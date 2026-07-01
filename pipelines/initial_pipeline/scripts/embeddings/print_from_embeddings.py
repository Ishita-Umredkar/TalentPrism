import pickle
import pprint

with open(
    "data/test/embedded_candidates.pkl",
    "rb"
) as f:
    candidates = pickle.load(f)


def truncate_embeddings(obj):

    if isinstance(obj, dict):

        result = {}

        for k, v in obj.items():

            if k == "embedding":

                result[k] = {
                    "dimension": len(v),
                    "first_5_values": v[:5].tolist()
                }

            else:

                result[k] = truncate_embeddings(v)

        return result

    elif isinstance(obj, list):

        return [
            truncate_embeddings(x)
            for x in obj
        ]

    return obj


pprint.pp(
    truncate_embeddings(
        candidates[0]
    ),
    width=120
)