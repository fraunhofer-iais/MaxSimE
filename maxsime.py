from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Rectangle
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.autonotebook import tqdm

from colbert import Indexer, Searcher
from colbert.data import Collection
from colbert.infra import ColBERTConfig, Run, RunConfig


class MaxSimE:
    dataroot = Path("~/data/LOTTE").expanduser()
    dataset = "pooled"
    datasplit = "dev"
    nbits = 2  # encode each dimension with 2 bits
    doc_maxlen = 300  # truncate passages at 300 tokens
    checkpoint = Path("~/models/colbertv2.0").expanduser()
    index_name = f"{dataset}.{datasplit}.{nbits}bits"

    def __init__(
        self,
        collection: Collection,
        model_type: str = "colbert",
        model: str = "colbert",
        reindex: bool = False,
        query_maxlen=512,
    ):
        self.collection = collection
        self.query_maxlen = query_maxlen
        self.type = model_type

        match model_type:
            case "bi-encoder":
                # load classic BERT models with SentenceTransformer library (mean-pooling will be added)
                self.model = SentenceTransformer(model)

            case "cross-encoder":
                self.model = CrossEncoder(model)
                self.model.model.cuda()

            case "colbert":
                # load msmarco-pretrained colbertv2 based searcher
                with Run().context(
                    RunConfig(experiment="MaxSimE", nranks=2)
                ):  # nranks specifies the number of GPUs to use.
                    config = ColBERTConfig(
                        doc_maxlen=self.doc_maxlen,
                        nbits=self.nbits,
                        query_maxlen=self.query_maxlen,
                    )

                    if reindex:
                        # this initializes the index
                        indexer = Indexer(checkpoint=self.checkpoint, config=config)
                        # this loads the pretrained model, encodes collection and indexes the passages into faiss
                        indexer.index(
                            name=self.index_name,
                            collection=self.collection,
                            overwrite=True,
                        )

                    # colBERT index searcher
                    self.searcher = Searcher(index=self.index_name)

    def get_tokens(self, sequence, query: bool = False):
        match self.type:
            case "colbert":
                if query:
                    return self.searcher.checkpoint.query_tokenizer.tokenize(
                        [sequence], add_special_tokens=True
                    )[0][: self.query_maxlen]
                else:
                    return self.searcher.checkpoint.doc_tokenizer.tokenize(
                        [sequence], add_special_tokens=True
                    )[0][: self.query_maxlen]

            case "bi-encoder" | "cross-encoder":
                return self.model.tokenizer.convert_ids_to_tokens(
                    self.model.tokenizer.encode(sequence)
                )[: self.query_maxlen]

    def get_embeddings(self, sequence):
        match self.type:
            case "colbert":
                return self.searcher.checkpoint.queryFromText([sequence])[
                    : self.query_maxlen
                ]
            case "bi-encoder":
                return self.model.encode(sequence, output_value="token_embeddings")[
                    : self.query_maxlen
                ]
            case "cross-encoder":
                features = self.model.tokenizer(
                    [sequence], return_tensors="pt", truncation=True
                )
                features = {k: f.cuda() for k, f in features.items()}
                output = self.model.model(**features, output_hidden_states=True)
                return output.hidden_states[-1][0][: self.query_maxlen].detach()

    def get_scores(self, Q, D):
        if self.type == "colbert":
            return torch.mm(Q[0].float(), D[0].permute(1, 0).float())
        else:
            return torch.nn.functional.normalize(Q).mm(
                torch.nn.functional.normalize(D).permute(1, 0)
            )

    def explain_match(self, query, doc, strip_special: bool = True):
        # tokenize
        query_tokens = self.get_tokens(query, query=True)
        doc_tokens = self.get_tokens(doc)

        # optionally strip query mask tokens
        special_tokens = ["[MASK]", "[SEP]", "[Q]"]
        non_special_token_ids = [
            i for i, t in enumerate(query_tokens) if t not in special_tokens
        ]

        # get embeddings from model
        Q = self.get_embeddings(query).cpu()
        D = self.get_embeddings(doc).cpu()

        # compute cosine similarity,
        scores = self.get_scores(Q, D)
        similarity = scores.max(-1)[0].sum()
        # only select non [MASK] query tokens
        if strip_special:
            query_tokens = [query_tokens[i] for i in non_special_token_ids]
            scores = scores[non_special_token_ids]
        # compute max sim candidates
        maxsim_ids = scores.argmax(1).unique()
        # shrink score matrix to candidates
        scores = scores[:, maxsim_ids]
        # strip document tokens that aren't selected by MaxSim
        doc_tokens = [
            doc_tokens[i] if i < len(doc_tokens) else "[PAD]" for i in maxsim_ids
        ]

        # put everything in one dataframe
        return (
            scores,
            similarity,
            doc_tokens,
            pd.DataFrame.from_dict(
                {
                    "query_token": query_tokens,
                    "doc_token": [doc_tokens[i] for i in scores.argmax(-1)],
                    "score": scores.max(-1)[0],
                }
            ).set_index("query_token"),
        )


def align_explanations(expl1: pd.DataFrame, expl2: pd.DataFrame):
    # join explanations
    aligned = expl1.join(expl2, lsuffix="_1", rsuffix="_2", how="outer")
    # drop the [Q] token
    # return aligned.drop(index="[Q]")
    return aligned


def fidelity(expl1: pd.DataFrame, expl2: pd.DataFrame):
    expl = align_explanations(expl1, expl2)

    token_precision = (
        expl["doc_token_2"].apply(lambda t: t in expl["doc_token_1"]).mean()
    )
    match_accuracy = (
        expl[["doc_token_1", "doc_token_2"]]
        .apply(lambda row: row[0] == row[1], axis=1)
        .mean()
    )
    spearman = expl[["score_1", "score_2"]].corr(method="spearman")["score_1"][
        "score_2"
    ]
    pearson = expl[["score_1", "score_2"]].corr(method="pearson")["score_1"]["score_2"]

    return token_precision, match_accuracy, spearman, pearson


def evaluate(mxsm1, mxsm2, queries, documents):
    doc_idx = 0
    token_precision, match_accuracy, spearman, pearson = [
        *zip(
            *[
                fidelity(
                    mxsm1.explain_match(query[1], doc)[-1],
                    mxsm2.explain_match(query[1], doc)[-1],
                )
                for query, doc in tqdm(zip(queries, documents), total=len(queries))
            ]
        )
    ]

    return (
        torch.Tensor(token_precision),
        torch.Tensor(match_accuracy),
        torch.Tensor(spearman),
        torch.Tensor(pearson),
    )


def visualize_match(scores, similarity, doc_tokens, expl, annot=False, ax=None):
    if ax is None:
        # define new figure
        plt.figure(
            figsize=(len(doc_tokens) // 1.5, len(query_tokens) // 2), tight_layout=True
        )
        ax = plt.gca()

    ax.set_title(f"MaxSim scores; score={similarity:.2f}")

    # heatmap of the token similarities
    sns.heatmap(
        scores,
        yticklabels=expl.index,
        xticklabels=doc_tokens,
        square=False,
        ax=ax,
        annot=annot,
        fmt=".2f",
        cbar=False,
        cmap="gray",
    )

    # highlight MaxSim matches
    for i, j in enumerate(scores.argmax(1)):
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="blue", lw=3))


def visualize_correlation(
    expl,
    text: bool = True,
    rotate_xticks: int = 0,
    labelsize="medium",
    labelweight="semibold",
    col_names=["score_1", "score_2"],
):
    # sort by score value
    expl = expl.sort_values(by=col_names, ascending=False)
    # plot
    scp = sns.scatterplot(expl)
    # rotate xticks
    scp.axes.tick_params(axis="x", rotation=rotate_xticks)
    scp.axes.set(ylabel="$cos$ similarity")

    if text:
        # add labels text
        for i in range(expl.shape[0]):
            if not np.isnan(expl[col_names[0]][i]):
                scp.text(
                    i + 0.01,
                    expl[col_names[0]][i] - 0.02,
                    expl["doc_token_1"][i],
                    horizontalalignment="left",
                    size=labelsize,
                    color="blue",
                    weight=labelweight,
                )
                scp.text(
                    i + 0.01,
                    expl[col_names[1]][i] + 0.01,
                    expl["doc_token_2"][i],
                    horizontalalignment="left",
                    size=labelsize,
                    color="orange",
                    weight=labelweight,
                )
