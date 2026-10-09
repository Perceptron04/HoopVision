import os
import pickle


def save_stub(stub_path, object):
    """
    Save cached processing results to disk.
    """
    if stub_path is None:
        return

    directory = os.path.dirname(stub_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(stub_path, "wb") as file:
        pickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_stub(read_from_stub, stub_path):
    """
    Load cached processing results when requested and available.
    """
    if (
        not read_from_stub
        or stub_path is None
        or not os.path.isfile(stub_path)
    ):
        return None

    try:
        with open(stub_path, "rb") as file:
            return pickle.load(file)
    except (OSError, pickle.UnpicklingError, EOFError):
        # An unreadable or incomplete cache should be regenerated.
        return None