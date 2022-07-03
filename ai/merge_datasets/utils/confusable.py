from collections import defaultdict
import re


# confusables.txt should contain contents of
# http://www.unicode.org/Public/security/latest/confusables.txt

class Confusables:

    def __init__(self, confusables_filename):
        """
        Parse confusables file into a dict of arrays.
        """
        f = open(confusables_filename, 'r')
        confusables_dict = defaultdict(list)
        pattern = re.compile(r'(.) → (.)')
        for line in f:
            r = pattern.search(line)
            if r:
                fake = r.group(1)
                auth = r.group(2)
                confusables_dict[auth].append(fake)
        self.confusables_dict = confusables_dict

    def expand_char_to_confusables(self, c):
        if c in self.confusables_dict:
            return '[{}{}]'.format(re.escape(c), re.escape("".join(self.confusables_dict[c])))
        else:
            return c

    def confusables_regex(self, pattern, letter_test_function=None):
        """
        Return string with each letter replaced with character class that
        matches the letter and any character that might be confused
        with it.
        """
        new = ""
        for c in pattern:
            if ((not letter_test_function) or
                    (letter_test_function and letter_test_function(c))):
                new += self.expand_char_to_confusables(c)
            else:
                new += c
        return new


import unicodedata
unicodedata.normalize('NFKD', '𝓗℮𝐥1೦𝗵𝗲𝗹𝗹𝗼')