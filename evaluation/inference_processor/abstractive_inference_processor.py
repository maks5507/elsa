#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

from elsa import Elsa
from pathlib import Path
import os
import pickle 

from worker_compose import noexcept
from torch.utils.data import Dataset, DataLoader

class AbstractiveSummaryDataset(Dataset):
    def __init__(self, sentence_score_path, basenames):
        super().__init__()
        self.sentence_score_paths = [
            f'{sentence_score_path}/{basename}.pkl'
            for basename in basenames
        ]

    def __getitem__(self, idx):
        with open(self.sentence_score_paths[idx], 'rb') as f:
            obj = pickle.load(f)

        basename = str(Path(self.sentence_score_paths[idx]).stem) 
        sentences = obj["sentences"]
        sentences_scores = obj["sentences_scores"]
        return (basename, sentences, sentences_scores)

    def __len__(self):
        return len(self.sentence_score_paths)

def collate_fn(batch):
    return tuple(zip(*batch))

class AbstractiveInferenceProcessor:
    def __init__(self, log, sentence_score_path, save_path, elsa_params):
        self.log = log
        self.sentence_score_path = sentence_score_path
        self.save_path = save_path
        self.elsa = Elsa(**elsa_params)

    @noexcept(default_value=None)
    def run(self, paths, batch_size, model_params):
        basenames = [str(Path(path).stem) for path in paths]
        dataset = AbstractiveSummaryDataset(self.sentence_score_path, basenames)
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        for batch_basename, batch_sentences, batch_sentence_score in data_loader:
            batch_summary = self.elsa.abstractive_summarize(
                batch_sentences, batch_sentence_score, **model_params
            )
            for basename, summary in zip(batch_basename, batch_summary):
                output_path = f'{self.save_path}/{basename}.summary'
                with open(output_path, 'w') as f_out:
                    f_out.write(summary+'\n')
