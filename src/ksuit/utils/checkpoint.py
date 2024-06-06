import functools
import math
import os
import re


@functools.total_ordering
class Checkpoint:
    @staticmethod
    def create(kwargs):
        if kwargs is None:
            return None
        if isinstance(kwargs, Checkpoint):
            return kwargs
        return Checkpoint(**kwargs)

    def __init__(self, epoch=None, update=None, sample=None):
        self.epoch = epoch
        self.update = update
        self.sample = sample

    def copy(self):
        return Checkpoint(epoch=self.epoch, update=self.update, sample=self.sample)

    @property
    def specified_properties_count(self):
        return sum([self.epoch is not None, self.update is not None, self.sample is not None])

    @property
    def is_fully_specified(self):
        return self.specified_properties_count == 3

    @property
    def is_minimally_specified(self):
        return self.specified_properties_count == 1

    def get_n_equal_properties(self, other):
        return sum([self.epoch == other.epoch, self.update == other.update, self.sample == other.sample])

    def to_fully_specified(self, updates_per_epoch, effective_batch_size):
        if self.is_fully_specified:
            return Checkpoint(self.epoch, self.update, self.sample)
        assert self.is_minimally_specified
        if self.update is not None:
            total_updates = self.update
        elif self.epoch is not None:
            total_updates = updates_per_epoch * self.epoch
        else:
            total_updates = int(self.sample / effective_batch_size)
        return Checkpoint(
            epoch=int(total_updates / updates_per_epoch),
            update=total_updates,
            sample=total_updates * effective_batch_size,
        )

    def scale(self, factor, updates_per_epoch, effective_batch_size, floor):
        # convert to updates
        if self.update is not None:
            updates = self.update
        else:
            if self.epoch is not None:
                updates = self.epoch * updates_per_epoch
            elif self.sample is not None:
                updates = self.sample / effective_batch_size
            else:
                raise NotImplementedError
        # scale
        updates *= factor
        # floor or ceil
        if floor:
            updates = int(updates)
        else:
            updates = math.ceil(updates)
        # convert to original specification
        ckpt = Checkpoint()
        if self.epoch is not None:
            ckpt.epoch = int(updates / updates_per_epoch)
        if self.update is not None:
            ckpt.update = updates
        if self.sample is not None:
            ckpt.sample = updates * effective_batch_size
        return ckpt

    def __eq__(self, other):
        return self.epoch == other.epoch and self.update == other.update and self.sample == other.sample

    def __hash__(self):
        return hash((self.epoch, self.update, self.sample))

    def __ge__(self, other):
        assert self.has_same_specified_properties(other)
        if self.epoch is not None and other.epoch is not None:
            if self.epoch < other.epoch: return False
        if self.update is not None and other.update is not None:
            if self.update < other.update: return False
        if self.sample is not None and other.sample is not None:
            if self.sample < other.sample: return False
        return True

    def has_same_specified_properties(self, other):
        if not ((self.epoch is None) == (other.epoch is None)): return False
        if not ((self.update is None) == (other.update is None)): return False
        if not ((self.sample is None) == (other.sample is None)): return False
        return True

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_minimally_specified:
            if self.epoch is not None: return f"Epoch {self.epoch}"
            if self.update is not None: return f"Update {self.update}"
            if self.sample is not None: return f"Sample {self.sample}"
        if isinstance(self.epoch, float):
            epoch_str = str(int(self.epoch))
        else:
            epoch_str = str(self.epoch)
        return f"E{epoch_str}_U{self.update}_S{self.sample}"

    @staticmethod
    def from_checkpoint_string(checkpoint_string):
        matches = re.findall("E(\\d*)_U(\\d*)_S(\\d*)", checkpoint_string)
        assert len(matches) == 1
        epoch_str, update_str, sample_str = matches[0]
        return Checkpoint(epoch=int(epoch_str), update=int(update_str), sample=int(sample_str))

    @staticmethod
    def contains_checkpoint_string(source):
        matches = re.findall("E\\d*_U\\d*_S\\d*", source)
        return len(matches) > 0

    @staticmethod
    def find_checkpoint_string(source):
        matches = re.findall("E\\d*_U\\d*_S\\d*", source)
        assert len(matches) == 1
        return matches[0]

    @staticmethod
    def from_filename(fname):
        assert Checkpoint.contains_checkpoint_string(fname)
        ckpt_str = Checkpoint.find_checkpoint_string(fname)
        return Checkpoint.from_checkpoint_string(ckpt_str)

    @staticmethod
    def to_fully_specified_from_fnames(ckpt_folder, ckpt, prefix=None, suffix=None):
        assert ckpt.is_fully_specified or ckpt.is_minimally_specified
        for f in os.listdir(ckpt_folder):
            # filter irrelevant files
            if prefix is not None and not f.startswith(prefix):
                continue
            if suffix is not None and not f.endswith(suffix):
                continue
            if not Checkpoint.contains_checkpoint_string(f):
                continue
            # extract Checkpoint object from filename
            ckpt_from_fname = Checkpoint.from_checkpoint_string(Checkpoint.find_checkpoint_string(f))
            # remove unnecesary properties for comparison (e.g. Checkpoint(epoch=5, update=12, samples=123) -->
            # Checkpoint(epoch=5) if checkpoint=Checkpoint(epoch=123))
            if ckpt_from_fname.to_target_specification(ckpt) == ckpt:
                return ckpt_from_fname
        raise FileNotFoundError(
            f"no checkpoint file found (folder='{ckpt_folder}' checkpoint='{ckpt}' "
            f"prefix='{prefix}' suffix='{suffix}')"
        )

    def to_target_specification(self, target):
        """
        removes all overly specified properties of self (depending on the specified properties of target)
        e.g.
        self=Checkpoint(epoch=6, update=12, sample=123)
        target=Checkpoint(epoch=5)
        returns a new Checkpoint(epoch=6)
        """
        assert target.specified_properties_count <= self.specified_properties_count
        kwargs = {}
        if target.epoch is not None:
            kwargs["epoch"] = self.epoch
        if target.update is not None:
            kwargs["update"] = self.update
        if target.sample is not None:
            kwargs["sample"] = self.sample
        return Checkpoint(**kwargs)

    def __add__(self, other):
        assert self.has_same_specified_properties(other)
        return Checkpoint(self.epoch + other.epoch, self.update + other.update, self.sample + other.sample)

    def __sub__(self, other):
        assert self.has_same_specified_properties(other)
        epoch = self.epoch - other.epoch if self.epoch is not None else None
        update = self.update - other.update if self.update is not None else None
        sample = self.sample - other.sample if self.sample is not None else None
        return Checkpoint(epoch, update, sample)

    def __iter__(self):
        # proxy for casting to dict
        # https://stackoverflow.com/questions/35282222/in-python-how-do-i-cast-a-class-object-to-a-dict
        if self.epoch is not None:
            yield "epoch", self.epoch
        if self.update is not None:
            yield "update", self.update
        if self.sample is not None:
            yield "sample", self.sample
