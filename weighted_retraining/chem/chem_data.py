""" Code for chem datasets """

import pickle
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# My imports
from weighted_retraining.chem.chem_utils import standardize_smiles, penalized_logP, logP_func, SAS_func, qed_func, tpsa_func, num_rb_func, NP_score_func, activity_func

# My imports
from weighted_retraining.chem.jtnn import MolTreeFolder, MolTreeDataset, Vocab, MolTree

NUM_WORKERS = 4

##################################################
# Data preprocessing code
##################################################
def get_vocab_from_tree(tree: MolTree):
    cset = set()
    for c in tree.nodes:
        cset.add(c.smiles)
    return cset


def get_vocab_from_smiles(smiles):
    """ Get the set of all vocab items for a given smiles """
    mol = MolTree(smiles)
    return get_vocab_from_tree(mol)


def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


##################################################
# Pytorch lightning data classes
##################################################
class WeightedMolTreeFolder(MolTreeFolder):
    """ special weighted mol tree folder """

    def __init__(self, prop,prop2,max_flag, property_dict,property_dict2, data_weighter,init_data, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store all the underlying data
        self.data = []
        for idx in range(len(self.data_files)):
            data = self._load_data_file(idx)
            self.data += data
            del data
            
            
        self.data_weighter = data_weighter

        self.property = prop
        if self.property == "pen_logP":
            self.prop_func = penalized_logP
        elif self.property == "logP":
            self.prop_func = logP_func
        elif self.property == "SAS":
            self.prop_func = SAS_func
        elif self.property == "qed":
            self.prop_func = qed_func
        elif self.property == "TPSA":
            self.prop_func = tpsa_func
        elif self.property == "Num_rot_bonds":
            self.prop_func = num_rb_func
        elif self.property == "NP_score":
            self.prop_func = NP_score_func
        elif self.property == "DRD2":
            self.prop_func = activity_func
        else:
            raise NotImplementedError(self.property)
        
        self.property2 = prop2
        if self.property2 == "SAS":
            self.prop_func2 = SAS_func
        elif self.property2 == "NP_score":
            self.prop_func2 = NP_score_func
        elif self.property2 == "DRD2":
            self.prop_func2 = activity_func
        # else:
        #     raise NotImplementedError(self.property2)
        
        self._set_data_properties(property_dict,property_dict2,max_flag,init_data)

    def _set_data_properties(self, property_dict,property_dict2,max_flag, init_data):
        """ Set various properties from the dataset """

        
        
        # if init_data!=0:
        #     idx_shuffle = np.random.choice(
        #         len(self.data),
        #         size=int(init_data*len(self.data)),
        #         replace=False
        #         )
        #     self.data= [self.data[i] for i in idx_shuffle]
            
        # Extract smiles from the data
        self.smiles = [t.smiles for t in self.data]    
        self.canonic_smiles = list(map(standardize_smiles, self.smiles))

        # Set the data length
        self._len = len(self.data) // self.batch_size

        # Calculate any missing properties
        if not set(self.canonic_smiles).issubset(set(property_dict.keys())):
            for s in tqdm(
                set(self.canonic_smiles) - set(property_dict), desc="calc properties"
            ):
                property_dict[s] = self.prop_func(s)
                
        if len(property_dict2)!=0:
            if not set(self.canonic_smiles).issubset(set(property_dict2.keys())):
                for s in tqdm(
                    set(self.canonic_smiles) - set(property_dict2), desc="calc properties"
                ):
                    property_dict2[s] = self.prop_func2(s)

        # Randomly check that the properties match the ones calculated
        # Check first few, random few, then last few
        max_check_size = min(10, len(self.data))
        prop_check_idxs = list(
            np.random.choice(
                len(self.canonic_smiles), size=max_check_size, replace=False
            )
        )
        prop_check_idxs += list(range(max_check_size)) + list(range(-max_check_size, 0))
        prop_check_idxs = sorted(list(set(prop_check_idxs)))
        for i in prop_check_idxs:
            s = self.canonic_smiles[i]
            assert np.isclose(
                self.prop_func(s), property_dict[s], rtol=1e-3, atol=1e-4
            ), f"score for smiles {s} doesn't match property dict for property {self.property}"



        # Finally, set properties attribute!
        if len(property_dict2)==0:
            self.data_properties = np.array([property_dict[s] for s in self.canonic_smiles])
            # Calculate weights (via maximization)
            self.weights = self.data_weighter.weighting_function(self.data_properties)
        else:
            self.data_properties = np.array([[property_dict[s],property_dict2[s]] for s in self.canonic_smiles])
            self.weights = self.data_weighter.weighting_function(self.data_properties, max_flag=max_flag)
        
        
        if init_data!=0:
            weighted_idx_shuffle = np.random.choice(
                len(self.weights),
                size=int(init_data*len(self.weights)),
                replace=False,
                p=self.weights / self.weights.sum(),
                )
            # self.weights= np.array([self.weights[i] for i in weighted_idx_shuffle])
            self.data=[self.data[i] for i in weighted_idx_shuffle]
            self.canonic_smiles= [self.canonic_smiles[i] for i in weighted_idx_shuffle]
            self.data_properties=np.array([self.data_properties[i] for i in weighted_idx_shuffle])
            self._len = len(self.data) // self.batch_size
            
            if len(property_dict2)==0:
                self.weights = self.data_weighter.weighting_function(self.data_properties)
            else:
                self.weights = self.data_weighter.weighting_function(self.data_properties, max_flag=max_flag)
        

        # Sanity check
        assert len(self.data) == len(self.data_properties) == len(self.weights)

    def __len__(self):
        return self._len

    def __iter__(self):
        """ iterate over the dataset with weighted choice """

        # Shuffle the data in a weighted way
        weighted_idx_shuffle = np.random.choice(
            len(self.weights),
            size=len(self.weights),
            replace=True,
            p=self.weights / self.weights.sum(),
        )

        # Make batches
        shuffled_data = [self.data[i] for i in weighted_idx_shuffle]
        batches = [
            shuffled_data[i : i + self.batch_size]
            for i in range(0, len(shuffled_data), self.batch_size)
        ]
        if len(batches[-1]) < self.batch_size:
            batches.pop()

        dataset = MolTreeDataset(batches, self.vocab, self.assm)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x[0],
        )
        for b in dataloader:
            yield b
        del dataset, dataloader, shuffled_data


class WeightedJTNNDataset(pl.LightningDataModule):
    """ dataset with property weights. Needs to load all data into memory """

    def __init__(self, hparams, data_weighter):
        super().__init__()
        self.train_path = hparams.train_path
        self.val_path = hparams.val_path
        self.vocab_file = hparams.vocab_file
        self.batch_size = hparams.batch_size
        self.property = hparams.property
        self.property_file = hparams.property_file
        self.property2 = hparams.property2
        self.property2_file = hparams.property2_file
        self.data_weighter = data_weighter

    @staticmethod
    def add_model_specific_args(parent_parser):
        data_group = parent_parser.add_argument_group(title="data")
        data_group.add_argument("--train_path", required=True)
        data_group.add_argument("--val_path", required=False, default=None)
        data_group.add_argument("--vocab_file", required=True)
        data_group.add_argument("--batch_size", type=int, default=32)
        data_group.add_argument(
            "--property", type=str, choices=["pen_logP","logP","SAS", "qed", "TPSA","Num_rot_bonds", "NP_score", "DRD2"], default="logP"
        )
        data_group.add_argument(
            "--property2", type=str, choices=["pen_logP","logP","SAS", "qed", "TPSA","Num_rot_bonds","NP_score", "DRD2"], default=None
        )
        data_group.add_argument(
            "--property_file",
            type=str,
            default=None,
            help="dictionary file mapping smiles to properties. Optional but recommended",
        )
        data_group.add_argument(
            "--property2_file",
            type=str,
            default=None,
            help="dictionary file mapping for second property",
        )
        return parent_parser

    def setup(self, stage):

        # Create vocab
        with open(self.vocab_file) as f:
            self.vocab = Vocab([x.strip() for x in f.readlines()])

        # Read in properties
        if self.property_file is None:
            property_dict = dict()
        else:
            with open(self.property_file, "rb") as f:
                property_dict = pickle.load(f)
        
        if self.property2_file is None:
            property_dict2 = dict()
        else:
            with open(self.property2_file, "rb") as f:
                property_dict2 = pickle.load(f)
        
        
        self.train_dataset = WeightedMolTreeFolder(
            self.property,self.property2,self.flag,
            property_dict, property_dict2,
            self.data_weighter,self.all_new,
            self.train_path,
            self.vocab,
            self.batch_size, 
            num_workers=NUM_WORKERS,
        )

        # Val dataset, if given
        if self.val_path is None:
            self.val_dataset = None
        else:
            self.val_dataset = WeightedMolTreeFolder(
                self.property,self.property2,self.flag,
                property_dict, property_dict2,
                self.data_weighter,0,
                self.val_path,
                self.vocab,
                self.batch_size, 
                num_workers=NUM_WORKERS,
            )

    def append_train_data(self, smiles_new, z_prop):
        dset = self.train_dataset
        max_flag=self.flag
        assert len(smiles_new) == len(z_prop)

        # Check which smiles need to be added!
        can_smiles_set = set(dset.canonic_smiles)
        if self.property2 is None:
            prop_dict = {s: p for s, p in zip(dset.canonic_smiles, dset.data_properties)}
            prop_dict2= dict()
        else:
            prop_dict = {s: p for s, p in zip(dset.canonic_smiles, dset.data_properties[:,0])}
            prop_dict2 = {s: p for s, p in zip(dset.canonic_smiles, dset.data_properties[:,1])}

        # Total vocabulary set
        vocab_set = set(self.vocab.vocab)

        # Go through and do the addition
        s_add = []
        data_to_add = []
        props_to_add = []
        for s, prop in zip(smiles_new, z_prop):
            if s is None:
                continue
            s_std = standardize_smiles(s)
            if s_std not in can_smiles_set:  # only add new smiles

                # tensorize data
                tree_tensor = tensorize(s_std)

                # Make sure satisfies vocab check
                v_set = get_vocab_from_tree(tree_tensor)
                if v_set <= vocab_set:

                    # Add to appropriate trackers
                    can_smiles_set.add(s_std)
                    s_add.append(s_std)
                    data_to_add.append(tree_tensor)
                    props_to_add.append(prop)

                    # Update property dict for later
                    if self.property2 is None:
                        prop_dict[s_std] = prop
                    else:
                        prop_dict[s_std] = prop[0]
                        prop_dict2[s_std] = prop[1]
        props_to_add = np.array(props_to_add)

        # Either add or replace the data, depending on the mode
        if self.data_weighter.weight_type == "fb":

            # Find top quantile
            cutoff = np.quantile(props_to_add, self.data_weighter.weight_quantile)
            indices_to_add = props_to_add >= cutoff

            # Filter all but top quantile
            s_add = [s_add[i] for i, ok in enumerate(indices_to_add) if ok]
            data_to_add = [data_to_add[i] for i, ok in enumerate(indices_to_add) if ok]
            props_to_add = props_to_add[indices_to_add]
            assert len(s_add) == len(data_to_add) == len(props_to_add)

            # Replace the first few data points
            dset.data = dset.data[len(data_to_add) :] + data_to_add

        else:
            dset.data += data_to_add

        # Now recalcuate weights/etc
        
        dset._set_data_properties(prop_dict,prop_dict2,max_flag,0)
        

        # Return what was successfully added to the dataset
        return s_add, props_to_add

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=lambda x: [x], batch_size=None)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset, collate_fn=lambda x: [x], batch_size=None
            )
