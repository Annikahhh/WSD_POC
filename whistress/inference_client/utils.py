import torch
from transformers import WhisperConfig
import librosa
import numpy as np
import pathlib
from torch.nn import functional as F
from ..model import WhiStress, WhiStressPhn, WhiStressPhnIa
import os
import json


PATH_TO_WEIGHTS = pathlib.Path(__file__).parent.parent / "weights"

def save_model_parts(model, save_dir, metadata):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.classifier.state_dict(), save_dir / "classifier.pt")
    torch.save(model.additional_decoder_block.state_dict(), save_dir / "additional_decoder_block.pt")
    metadata["path_to_weights"] = str(save_dir)
    
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def get_loaded_model(device="cuda", metadata=None):
    if metadata is None:
        whisper_model_name = f"openai/whisper-small.en"
        layer_for_head=9
        model_type = "WhiStress"
    else:
        whisper_model_name = metadata["whisper_tag"]
        layer_for_head=metadata["layer_for_head"]
        model_type = metadata["model_type"]
    
    whisper_config = WhisperConfig()
    
    if model_type == "WhiStress":
        print("Inference WhiStress")
        whistress_model = WhiStress(
            config=whisper_config, 
            layer_for_head=layer_for_head, 
            whisper_backbone_name=whisper_model_name
        ).to(device)
    elif model_type == "WhiStressPhn":
        print("Inference WhiStressPhn")
        whistress_model = WhiStressPhn(
            config=whisper_config, 
            layer_for_head=layer_for_head, 
            whisper_backbone_name=whisper_model_name,
            num_phones=39, 
        ).to(device)
    elif model_type == "WhiStressPhnIa":
        print("Inference WhiStressPhnIa")
        whistress_model = WhiStressPhnIa(
            config=whisper_config, 
            layer_for_head=layer_for_head, 
            whisper_backbone_name=whisper_model_name,
            num_phones=39, 
        ).to(device)
    else:
        raise ValueError(f"model_type {model_type} hasn't been implemented yet.")
    
    whistress_model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]
    
    if metadata is None:
        whistress_model.load_model(PATH_TO_WEIGHTS)
    else:
        if model_type in ["WhiStressPhn", "WhiStressPhnIa"]:
            print("Load All Weights")
            whistress_model.load_state_dict(torch.load(os.path.join(metadata["path_to_weights"], "model.pt")))
        else:
            print("Load PT Weights (load_model)")
            whistress_model.load_model(metadata["path_to_weights"])
    
    whistress_model.to(device)
    whistress_model.eval()
    return whistress_model


def get_word_emphasis_pairs(
    transcription_preds,
    emphasis_preds,
    emphasis_scores,
    processor,
    filter_special_tokens=True,
):
    emphasis_preds_list = emphasis_preds.tolist()
    emphasis_scores_list = emphasis_scores.tolist()

    transcription_preds_words = [
        processor.tokenizer.decode([i], skip_special_tokens=False)
        for i in transcription_preds
    ]

    if filter_special_tokens:
        special_tokens_indices = [
            i
            for i, x in enumerate(transcription_preds)
            if x in processor.tokenizer.all_special_ids
        ]

        emphasis_preds_list = [
            x for i, x in enumerate(emphasis_preds_list)
            if i not in special_tokens_indices
        ]

        emphasis_scores_list = [
            x for i, x in enumerate(emphasis_scores_list)
            if i not in special_tokens_indices
        ]

        transcription_preds_words = [
            x for i, x in enumerate(transcription_preds_words)
            if i not in special_tokens_indices
        ]

    return list(
        zip(
            transcription_preds_words,
            emphasis_preds_list,
            emphasis_scores_list
        )
    )


def inference_from_audio(
    audio: np.ndarray,
    model: WhiStress,
    device: str
):
    input_features = model.processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )["input_features"]

    out_model = model.generate_dual(
        input_features=input_features.to(device)
    )

    emphasis_probs = F.softmax(out_model.logits, dim=-1)

    # 類別 1（重讀）的分數，範圍約為 0～1
    stress_scores = emphasis_probs[:, :, 1]

    # 原始模型的二元類別預測
    emphasis_preds = torch.argmax(emphasis_probs, dim=-1)

    # 與 Whisper 輸出 token 對齊
    emphasis_preds_right_shifted = torch.cat(
        (emphasis_preds[:, -1:], emphasis_preds[:, :-1]),
        dim=1
    )

    stress_scores_right_shifted = torch.cat(
        (stress_scores[:, -1:], stress_scores[:, :-1]),
        dim=1
    )

    # 每筆結果格式：
    # (token文字, 原始重讀預測0/1, 重讀信心分數)
    word_emphasis_pairs = get_word_emphasis_pairs(
        out_model.preds[0],
        emphasis_preds_right_shifted[0],
        stress_scores_right_shifted[0],
        model.processor,
        filter_special_tokens=True,
    )

    return word_emphasis_pairs

def prepare_audio(audio, target_sr=16000):
    # resample to 16kHz
    sr = audio["sampling_rate"]
    y = audio["array"]
    y = np.array(y, dtype=float)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # Normalize the audio (scale to [-1, 1])
    y_resampled /= max(abs(y_resampled))
    return y_resampled


def merge_stressed_tokens(tokens_with_stress):
    merged = []

    current_word = ""
    current_stress = 0
    current_score = 0.0

    for token, stress, score in tokens_with_stress:
        if token.startswith(" ") or current_word == "":
            if current_word:
                merged.append(
                    (current_word, current_stress, current_score)
                )

            current_word = token
            current_stress = int(stress)
            current_score = float(score)
        else:
            current_word += token
            current_stress = max(current_stress, int(stress))
            current_score = max(current_score, float(score))

    if current_word:
        merged.append(
            (current_word, current_stress, current_score)
        )

    return merged


def inference_from_audio_and_transcription(
    audio: np.ndarray,
    transcription,
    model: WhiStress,
    device: str,
    phone_ids=None,
    token_pos_ids=None,
):
    input_features = model.processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )["input_features"]

    input_ids = model.processor.tokenizer(
        transcription,
        return_tensors="pt",
        truncation=True,
        max_length=50,
    )["input_ids"]

    out_model = model(
        input_features=input_features.to(device),
        decoder_input_ids=input_ids.to(device),
        phone_ids=phone_ids,
        token_pos_ids=token_pos_ids
    )

    emphasis_probs = F.softmax(out_model.logits, dim=-1)

    # 類別 1：重讀的信心分數
    stress_scores = emphasis_probs[:, :, 1]

    # 原始二元分類結果
    emphasis_preds = torch.argmax(emphasis_probs, dim=-1)

    # 與 decoder token 對齊
    emphasis_preds_right_shifted = torch.cat(
        (emphasis_preds[:, -1:], emphasis_preds[:, :-1]),
        dim=1
    )

    stress_scores_right_shifted = torch.cat(
        (stress_scores[:, -1:], stress_scores[:, :-1]),
        dim=1
    )

    # 回傳：(token文字, 預測0/1, 重讀信心分數)
    word_emphasis_pairs = get_word_emphasis_pairs(
        input_ids[0],
        emphasis_preds_right_shifted[0],
        stress_scores_right_shifted[0],
        model.processor,
        filter_special_tokens=True,
    )

    phone_stress_preds = out_model.phone_stress_preds

    return word_emphasis_pairs, phone_stress_preds

def scored_transcription(audio, model, strip_words=True, transcription: str = None, device="cuda", phone_ids=None, token_pos_ids=None):
    audio_arr = prepare_audio(audio)
    token_stress_pairs, phone_stress_preds = None, None
    if transcription: # if we want to use the ground truth transcription
        # TODO phone_ids & token_pos_ids
        token_stress_pairs, phone_stress_preds = inference_from_audio_and_transcription(audio_arr, transcription, model, device, phone_ids, token_pos_ids)
    else:
        token_stress_pairs = inference_from_audio(audio_arr, model, device)
    
    word_level_stress = merge_stressed_tokens(token_stress_pairs)

    if strip_words:
        word_level_stress = [
            (word.strip(), stress, confidence)
            for word, stress, confidence in word_level_stress
        ]

    return word_level_stress, phone_stress_preds
