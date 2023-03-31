import typing
import numpy as np


class Saveable:
    __save_parameters__ = []

    def __init__(self, **kwargs):
        for name in kwargs:
            if name in self.__save_parameters__:
                setattr(self, name, kwargs[name])

    def to_dict(self):
        data = {}
        for param in self.__save_parameters__:
            attribute = getattr(self, param, None)
            if attribute is not None:
                if getattr(attribute, "to_dict", None) is not None:
                    data[param] = getattr(attribute, "to_dict")()
                elif isinstance(attribute, list) and len(attribute) and (getattr(attribute[0], "to_dict", None) is not None or attribute[0] is None):
                    my_list = []
                    for attr in attribute:
                        value = attr
                        if getattr(attribute[0], "to_dict", None):
                            value = getattr(attr, "to_dict")()
                        if attr is None:
                            value = "__NONE__"
                        my_list.append(value)
                    data[param] = my_list
                elif attribute is None:
                    data[param] = "__NONE__"
                else:
                    data[param] = attribute
        return data

    def save(self, filename: str):
        data = self.to_dict()

        if filename.endswith("h5py") or filename.endswith("h5"):
            return dict_to_h5(filename, flatten_dict(data))

        #np.savez(filename, **data)
        np.lib.npyio._savez(filename, [], flatten_dict(data), True, allow_pickle=False)

    @classmethod
    def from_dict(cls, data_dict):
        types = typing.get_type_hints(cls)
        data = {}
        for name in data_dict:
            if isinstance(data_dict[name], np.ndarray) and len(data_dict[name].shape) == 0:
                data[name] = data_dict[name][()]
            else:
                data[name] = data_dict[name]
            if name in types:
                if getattr(types[name], "from_dict", None) is not None:
                    data[name] = types[name].from_dict(data[name])
                elif typing.get_origin(types[name]) is list:
                    if isinstance(data[name], dict):
                        data[name] = typing.get_args(types[name])[0].from_dict(data[name])
                    else:
                        data[name] = [None if d == "__NONE__" else typing.get_args(types[name])[0].from_dict(d) for d in data[name]]
        return cls(**data)

    @classmethod
    def load(cls, filename):
        if str(filename).endswith(".h5py") or str(filename).endswith(".h5"):
            import h5py
            data = h5py.File(filename, "a")
            result = cls.from_dict(unflatten_dict_h5(data))
        else:
            data = np.load(filename, allow_pickle=False)

            result = cls.from_dict(unflatten_dict(data))
        if getattr(result, 'on_load', None) is not None:
            getattr(result, 'on_load')(filename)
        return result


def flatten_dict(data):
    result = {}

    def print_content(data, prefix):
        if isinstance(data, list):  # and not isinstance(data[0], (int, float)):
            result[prefix] = "list"
            for name, d in enumerate(data):
                if d is None:
                    d = "__NONE__"
                print_content(d, f"{prefix}/{name}")
            return
        if isinstance(data, tuple):  # and not isinstance(data[0], (int, float)):
            result[prefix] = "tuple"
            for name, d in enumerate(data):
                if d is None:
                    d = "__NONE__"
                print_content(d, f"{prefix}/{name}")
            return
        if isinstance(data, dict):  # and not isinstance(data[0], (int, float)):
            result[prefix] = "dict"
            for name, d in data.items():
                if d is None:
                    d = "__NONE__"
                print_content(d, f"{prefix}/{name}")
            return
        result[prefix] = data

    for name, d in data.items():
        print_content(d, name)

    return result


def unflatten_dict(data):
    result = {}
    for name, item in data.items():
        if item.shape == ():
            item = item[()]
        if isinstance(item, str) and item == "list":
            item = []
        if isinstance(item, str) and item == "dict":
            item = {}
        if isinstance(item, str) and item == "tuple":
            item = ()

        names = name.split("/")

        hierarchy = [result]
        r = result
        for name in names[:-1]:
            try:
                r = r[name]
            except TypeError:
                r = r[int(name)]
            except KeyError:
                continue
            hierarchy.append(r)

        if isinstance(r, list):
            r += [item]
        elif isinstance(r, tuple):
            try:
                hierarchy[-2][names[-2]] = r + (item,)
            except TypeError:
                hierarchy[-2][int(names[-2])] = r + (item,)
        else:
            if isinstance(item, str) and item == "__NONE__":
                item = None
            r[names[-1]] = item

    return result


def dict_to_h5(filename, data):
    import h5py
    with h5py.File(filename, "w") as f:
        for key in data.keys():
            if isinstance(data[key], str):
                if str(data[key]) == "list" or str(data[key]) == "dict" or str(data[key]) == "tuple":
                    grp = f.create_group(key)
                    grp.attrs["type"] = str(data[key])
                    continue
                dset = f.create_dataset(key, data=str(data[key]))
            elif str(data[key].dtype).startswith("<U"):
                dset = f.create_dataset(key, data=str(data[key]))
            else:
                try:
                    f.create_dataset(key, data=data[key], compression="gzip")
                except TypeError:
                    f.create_dataset(key, data=data[key])


def unflatten_dict_h5(data):
    import h5py
    if isinstance(data, h5py.Group):
        if data.attrs.get("type") == "list":
            result = [unflatten_dict_h5(v) for v in data.values()]
        elif data.attrs.get("type") == "tuple":
            result = tuple([unflatten_dict_h5(v) for v in data.values()])
        else:
            result = {k: unflatten_dict_h5(v) for k, v in data.items()}
        return result
    else:
        if data.shape == ():
            try:
                return data.asstr()[()]
            except TypeError:
                return data[()]
        else:
            return data
