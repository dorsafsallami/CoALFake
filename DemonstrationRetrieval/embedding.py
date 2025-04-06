import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def get_embeddings(args, text):
    """
    Encode a list of text sentences into embeddings.
    
    Args:
        args: Contains parameters such as model name and device.
        text: List or pandas Series of text sentences.
    
    Returns:
        A numpy array of embeddings.
    """
    if not isinstance(text, list):
        text = text.tolist()

    model = SentenceTransformer(args.embedding_model_name_or_path)
    model.to(args.device)
    embeddings = []

    batch_size = 20
    for i in tqdm(range(0, len(text), batch_size), desc='Calculating embeddings'):
        batch_text = text[i:i + batch_size]
        if not batch_text:
            continue
        encoded_batch = model.encode(batch_text, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(encoded_batch)

    if embeddings:
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().numpy()

    return embeddings


def get_cosine_similarity(key: torch.Tensor, value: torch.Tensor):
    """
    Key, value are embeddings with same dimension.
    """
    key = F.normalize(key, dim=-1)
    value = F.normalize(value, dim=-1)
    cos_sim = torch.mm(key, value.transpose(0, 1))
    return cos_sim