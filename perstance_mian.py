import pandas as pd
import os
import json
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score
import time
import psutil  

def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return processor, model


def analyze_joint_personality(processor, model, text, image, target):
    joint_personality_prompt = (
        f"Analyze the following tweet, which contains both a text and an image, to describe in freeform the personality traits of the author who posted them."
        f"Focus on the emotions, tone, attitude, and visual style reflected in the content.\n\n"
        f"Consider what kind of person would write this text and post this image, especially in the context of the topic: '{target}'.\n"
        f"Base your analysis on:\n"
        f"- the language style, sentiment, diction tendency, tone and attitude of the text,\n"
        f"- the visual tone, composition, symbolism, and emotional cues in the image.\n\n"
        f"Text: {text}\n\n"
        f"Describe the personality of the person who posted this tweet."
    )
# body language or facial expression 
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": joint_personality_prompt}
            ]
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


def stance_detection(processor, model, text, image_path, target):
    image = Image.open(image_path).convert("RGB")
    
    joint_personality = analyze_joint_personality(processor, model, text, image, target) #Enable personality module
     
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        f"Description of the author's personality based on text and images: {joint_personality}\n" #Introducing personality descriptions
                        
                        #Optional when introducing GPT-4v's knowledge base
                        # f"Additionally, here is a reasoning passage related to the stance expressed by the content:\n"
                        # f"{auxiliary_reasoning}\n\n"

                        f"Analyze the given text and image to determine the stance towards the target: '{target}'.\n"
                        f"The stance should be categorized into one of the following classes:\n"
                        f"- 'Favor': If the text and image directly or indirectly expresses positive emotions towards the target (or expresses negative emotions towards the opposite target).\n"
                        f"- 'Against': If the text and image directly or indirectly expresses negative emotions towards the target (or expresses positive emotions towards the opposite target).\n"
                        f"- 'Neutral': If the text and image do not clearly support or oppose the target.\n\n"
                        f"Text: '{text}'\n"
                        f"Think step by step: first analyze what attitude the text expresses towards the '{target}', then analyze the attitude the image expresses towards the '{target}', and finally make a decision\n"
                        f"Provide only one of the three labels: 'Favor', 'Neutral', or 'Against'."

                        #Ablation of multiple CoT settings
                        # f"Think step by step: first analyze the attitude the text expresses towards the '{target}', and finally make a decision\n"
                        # f"Think step by step: first analyze the attitude the image expresses towards the '{target}', and finally make a decision\n"
                        # f"Think step by step.\n"
                    )
                }
            ]
        }
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    torch.cuda.empty_cache()
    return output_text.strip(), joint_personality


def evaluate_stance_detection(csv_paths, image_base_path, output_json_path):
    df = pd.read_csv(csv_paths, encoding="latin1")
    processor, model = load_model()
    true_labels = []
    pred_labels = []
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Stance Detection"):
        text = row["tweet_text"]
        image_path = f"{image_base_path}/{row['tweet_image']}"
        target = row["stance_target"]
        true_label = row["stance_label"]
        
        #Optional when introducing GPT-4v's knowledge base
        # auxiliary_reasoning = row.get("gpt4v_cot_response")
        
        pred_stance, joint_personality= stance_detection(processor, model, text, image_path, target)
        
        true_labels.append(true_label)
        pred_labels.append(pred_stance)
        results.append({
            "text": text,
            "image": row['tweet_image'],
            "target": target,
            "true_stance": true_label,
            "predicted_stance": pred_stance,
            "tweet_personality": joint_personality
        })
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    return acc, f1, recall

#Enable statistics on video memory usage and inference time, which can replace the above evaluation functions

# def evaluate_stance_detection(csv_paths, image_base_path, output_json_path):
#     df = pd.read_csv(csv_paths).head(200)
#     processor, model = load_model()

#     true_labels = []
#     pred_labels = []
#     results = []

#     per_sample_times = []
#     per_sample_cuda_peaks_mb = []
#     per_sample_rss_deltas_mb = []

#     proc = psutil.Process(os.getpid())
#     use_cuda = torch.cuda.is_available()

#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Stance Detection"):
#         text = row["tweet_text"]
#         image_path = f"{image_base_path}/{row['tweet_image']}"
#         target = row["stance_target"]
#         true_label = row["stance_label"]

#         if use_cuda:
#             torch.cuda.reset_peak_memory_stats()
#             torch.cuda.synchronize()
#         rss_before = proc.memory_info().rss  
#         start = time.perf_counter()

#         pred_stance, joint_personality = stance_detection(processor, model, text, image_path, target)

#         if use_cuda:
#             torch.cuda.synchronize()
#         elapsed = time.perf_counter() - start  

#         if use_cuda:
#             peak_cuda_bytes = torch.cuda.max_memory_allocated()
#             peak_cuda_mb = peak_cuda_bytes / (1024 ** 2)
#         else:
#             peak_cuda_mb = None

#         rss_after = proc.memory_info().rss
#         rss_delta_mb = (rss_after - rss_before) / (1024 ** 2)

#         per_sample_times.append(elapsed)
#         if peak_cuda_mb is not None:
#             per_sample_cuda_peaks_mb.append(peak_cuda_mb)
#         per_sample_rss_deltas_mb.append(rss_delta_mb)

#         true_labels.append(true_label)
#         pred_labels.append(pred_stance)
#         results.append({
#             "text": text,
#             "image": row['tweet_image'],
#             "target": target,
#             "true_stance": true_label,
#             "predicted_stance": pred_stance,
#             "tweet_personality": joint_personality,
#             "inference_time_sec": round(elapsed, 6),
#             "peak_cuda_memory_MB": round(peak_cuda_mb, 2) if peak_cuda_mb is not None else None,
#             "rss_memory_delta_MB": round(rss_delta_mb, 2)
#         })

#     acc = accuracy_score(true_labels, pred_labels)
#     f1 = f1_score(true_labels, pred_labels, average='macro')
#     recall = recall_score(true_labels, pred_labels, average='macro')

#     avg_time = sum(per_sample_times) / len(per_sample_times) if per_sample_times else 0.0
#     avg_cuda_peak = (sum(per_sample_cuda_peaks_mb) / len(per_sample_cuda_peaks_mb)) if per_sample_cuda_peaks_mb else None
#     avg_rss_delta = sum(per_sample_rss_deltas_mb) / len(per_sample_rss_deltas_mb) if per_sample_rss_deltas_mb else 0.0

#     summary = {
#         "accuracy": round(acc, 4),
#         "f1_macro": round(f1, 4),
#         "recall_macro": round(recall, 4),
#         "avg_inference_time_sec_per_sample": round(avg_time, 6),
#         "avg_peak_cuda_memory_MB_per_sample": round(avg_cuda_peak, 2) if avg_cuda_peak is not None else None,
#         "avg_rss_memory_delta_MB_per_sample": round(avg_rss_delta, 2)
#     }

#     os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
#     with open(output_json_path, "w", encoding="utf-8") as f:
#         json.dump({"summary": summary, "results": results}, f, indent=4, ensure_ascii=False)

#     print(f"Accuracy: {acc:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"Avg Time/sample: {avg_time:.6f}s")
#     if avg_cuda_peak is not None:
#         print(f"Avg Peak CUDA Mem/sample: {avg_cuda_peak:.2f} MB")
#     print(f"Avg RSS Delta/sample: {avg_rss_delta:.2f} MB")

#     return acc, f1, recall, summary

if __name__ == "__main__":
    # Paths to training, validation, and test CSV files
    csv_paths = "your_path/test.csv"
    image_base_path = "your_path"
    output_json_path = "your_path"

    evaluate_stance_detection(csv_paths, image_base_path, output_json_path) 