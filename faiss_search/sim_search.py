import os
import torch
from huggingface_hub import hf_hub_url
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel

# Corriger le problème OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


data_files = hf_hub_url(
    repo_id="lewtun/github-issues",
    filename='datasets-issues-with-comments.jsonl',
    repo_type='dataset'
)

issues_dataset = load_dataset('json', data_files=data_files, split='train')
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)

columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

issues_dataset.set_format("pandas")
df = issues_dataset[:]

comments_df = df.explode("comments", ignore_index=True)

comments_dataset = Dataset.from_pandas(comments_df)

comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)

comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }

comments_dataset = comments_dataset.map(concatenate_text)

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

# Code dupliqué supprimé - concatenate_text est déjà appliqué

device = torch.device("mps")
model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embedding = get_embeddings([comments_dataset["text"][0]])
print(f"Embedding shape: {embedding.shape}")

embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings([x["text"]]).detach().cpu().numpy()[0]}
)

embeddings_dataset.add_faiss_index(column="embeddings")

question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
print(f"Question embedding shape: {question_embedding.shape}")

scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding[0], k=5
)

# Afficher les résultats
print(f"\nQuestion: {question}")
print(f"Scores de similarité: {scores}")
print("\nRésultats les plus similaires:")
for i, (score, sample) in enumerate(zip(scores, samples["text"])):
    print(f"\n--- Résultat {i+1} (Score: {score:.4f}) ---")
    print(sample[:300] + "..." if len(sample) > 300 else sample)
