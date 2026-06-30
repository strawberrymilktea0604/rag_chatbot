import hashlib

def compute_file_hash(file):
    """
    Compute the MD5 hash of a file-like object.
    """
    hash_md5 = hashlib.md5()
    # Read the file in chunks to avoid memory issues
    for chunk in iter(lambda: file.read(4096), b""):
        hash_md5.update(chunk)
    file.seek(0)  # Reset the file pointer to the beginning after reading
    return hash_md5.hexdigest()