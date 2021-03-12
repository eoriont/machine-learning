import sys
sys.path.append('tests')
from otest import do_assert

def parse_line(line):
    entries = []
    entry_str = ""
    quote_symbol = None
    for char in line:
        if quote_symbol is None and char == ",":
            entries.append(entry_str)
            entry_str = ""
        else:
            entry_str += char
        if char in ("\"", "'"):
            if quote_symbol is None:
                quote_symbol = char
            elif quote_symbol == char:
                quote_symbol = None
    entries.append(entry_str)
    return entries

if __name__ == "__main__":
    line_1 = "1,0,3,'Braund, Mr. Owen Harris',male,22,1,0,A/5 21171,7.25,,S"
    do_assert("Parse line 1", parse_line(line_1),
    ['1', '0', '3', "'Braund, Mr. Owen Harris'", 'male', '22', '1', '0', 'A/5 21171', '7.25', '', 'S'])

    line_2 = '102,0,3,"Petroff, Mr. Pastcho (""Pentcho"")",male,,0,0,349215,7.8958,,S'
    do_assert("Parse line 2", parse_line(line_2),
    ['102', '0', '3', '"Petroff, Mr. Pastcho (""Pentcho"")"', 'male', '', '0', '0', '349215', '7.8958', '', 'S'])

    line_3 = '187,1,3,"O\'Brien, Mrs. Thomas (Johanna ""Hannah"" Godfrey)",female,,1,0,370365,15.5,,Q'
    do_assert("Parse line 3", parse_line(line_3),
    ['187', '1', '3', '"O\'Brien, Mrs. Thomas (Johanna ""Hannah"" Godfrey)"', 'female', '', '1', '0', '370365', '15.5', '', 'Q'])
