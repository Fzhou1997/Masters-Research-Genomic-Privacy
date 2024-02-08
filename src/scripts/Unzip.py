import os
import zipfile

OPENSNP_DIR = './datasets/OpenSNP'


def is_zip(path: str):
    """Check if a file is a ZIP archive by examining the local file header signature."""
    with open(path, 'rb') as f:
        bytes = f.read(4)
        return len(bytes) >= 4 and bytes[0] == 0x50 and bytes[1] == 0x4b and bytes[2] == 0x03 and bytes[3] == 0x04


def extract_zip(path):
    """Extract the genomic data stored in a zip file.

    Args:
        path (str): The path of the zip archive.
    """
    renamed_path = path[:-3] + 'zip'

    with zipfile.ZipFile(path, 'r') as zf:
        fname = zf.filelist[0].filename
        unpacked = open(os.path.join(OPENSNP_DIR, 'temp'), 'w')
        unpacked.write(zf.read(fname).decode('utf-8'))
        unpacked.close()

    os.rename(path, path[:-3] + 'zip')
    os.rename(os.path.join(OPENSNP_DIR, 'temp'), path)


if __name__ == '__main__':
    # iterate through every file in the dataset directory
    for fname in os.listdir(OPENSNP_DIR):
        fpath = os.path.join(OPENSNP_DIR, fname)

        # skip directories, we only care about zips hidden as text files
        if os.path.isdir(fpath):
            continue
        # skip actual zip files, ^^^^^ ditto
        if fpath.endswith('.zip'):
            continue

        # perform unzipping operation
        if is_zip(fpath):
            extract_zip(fpath)
