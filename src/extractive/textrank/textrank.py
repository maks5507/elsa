#
# Created by
#

from .. import FastTextWrapper


class Textrank:
    def __init__(self):
        pass

    def summarize(self, sentences: List[str], factor: int) -> List[Tuple(int, int)]:
        """
        Params:
            - factor: percent of sentences to keep in summary.
             Example: factor=0.5 restricts the summary volume to be less than 50% of the original text.
        returns: list of pairs (sentence index, score)
        """
        pass

