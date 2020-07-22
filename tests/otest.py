
colors = {
    'Black': '\u001b[30m',
    'Red': '\u001b[31m',
    'Green': '\u001b[32m',
    'Yellow': '\u001b[33m',
    'Blue': '\u001b[34m',
    'Magenta': '\u001b[35m',
    'Cyan': '\u001b[36m',
    'White': '\u001b[37m',
    'Reset': '\u001b[0m',
}


def do_assert(test_name, output, expected):
    assert output == expected, color_string(
        f"Test {test_name} failed: output {output} expected to be {expected}", 'Red')
    print(color_string(f"Test {test_name} PASSED!", 'Green'))


def assert_exception(test_name, func, *args):
    try:
        func(*args)
    except Exception:
        print(color_string(f'Test {test_name} PASSED!', 'Green'))
        return
    raise Exception(
        color_string(f"Test {test_name} failed because it didn't throw an exception!", 'Red'))


def color_string(s, col):
    return f"{colors[col]}{s}{colors['Reset']}"
