'''
File utilities for preprocessing
'''

def count_file_lines(filepath, chunk_size=1024*1024):
    '''
    Counts the number of lines in a file

    Args:
        filepath (str): The path to the file to count the lines of
        chunk_size (int): The size of the chunk to read in at a time

    Returns:
        int: The number of lines in the file
    '''
    count = 0
    with open(filepath, "rb") as f:
        leftover = b""
        while chunk := f.read(chunk_size):
            count += chunk.count(b"\n")
            leftover = chunk
        # if last chunk doesn’t end with newline, add 1
        if leftover and not leftover.endswith(b"\n"):
            count += 1
    return count

def count_file_line_whitespaces(filepath, chunk_size=1024*1024):
    '''
    Counts the number of lines with only whitespaces in a file

    Args:
        filepath (str): The path to the file to count the empty lines of
        chunk_size (int): The size of the chunk to read in at a time

    Returns:
        int: The number of empty lines in the file
    '''
    count = 0
    buffer = b""
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if not line.strip():
                    count += 1
    # check last line
    if not buffer.strip():
        count += 1
    return count
