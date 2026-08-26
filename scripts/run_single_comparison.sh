#!/usr/bin/env bash
# Train both Llama2 and PatchTST models for prediction window 2017-10-23
# Then call compare_llama2_vs_patchtst.py to generate comparison plot

set -euo pipefail

# ── GPU Setup ──────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1 # please use your own setting

# ── Common Parameters ──────────────────────────────────────────────────────────
ROOT_PATH="./dataset/"
DATA_PATH="NorthChina_diff.csv"
TEST_DATE="2017-10-23"
seq_len=52
pred_len=13
label_len=18
percent=100
train_epochs=64
itr=3

# ── Llama2 Specific Parameters ────────────────────────────────────────────────
llama2_model=Llama2
llama2_gpt_path="Model_from_HF/LLAMA2" # please use your own setting
llama2_batch_size=4
llama2_lr=1e-4
llama2_llama_layers=32
llama2_d_model=4096
llama2_n_heads=4
llama2_d_ff=4096

# ── PatchTST Specific Parameters ──────────────────────────────────────────────
patchtst_model=PatchTST
patchtst_batch_size=16
patchtst_lr=0.0001
patchtst_gpt_layer=6
patchtst_d_model=768
patchtst_n_heads=4
patchtst_d_ff=768

mkdir -p logs
mkdir -p figures

echo "========================================================"
echo " Starting comparison experiment: test_end_date=${TEST_DATE}"
echo "========================================================"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Train Llama2 model
# ═══════════════════════════════════════════════════════════════════════════════
echo "──────────────────────────────────────────────────────"
echo " [1/3] Training Llama2 model (${TEST_DATE})"
echo "──────────────────────────────────────────────────────"

llama2_model_id="flu_north_${llama2_model}_pl${pred_len}_${TEST_DATE}"

python main.py \
  --root_path     "${ROOT_PATH}" \
  --data_path     "${DATA_PATH}" \
  --model_id      "${llama2_model_id}" \
  --model         "${llama2_model}" \
  --gpt_path      "${llama2_gpt_path}" \
  --data          custom \
  --features      S \
  --target        positive_rate \
  --seq_len       "${seq_len}" \
  --label_len     "${label_len}" \
  --pred_len      "${pred_len}" \
  --test_end_date "${TEST_DATE}" \
  --batch_size    "${llama2_batch_size}" \
  --learning_rate "${llama2_lr}" \
  --train_epochs  "${train_epochs}" \
  --decay_fac     0.75 \
  --d_model       "${llama2_d_model}" \
  --n_heads       "${llama2_n_heads}" \
  --d_ff          "${llama2_d_ff}" \
  --freq          W \
  --patch_size    16 \
  --stride        2 \
  --percent       "${percent}" \
  --llama_layers  "${llama2_llama_layers}" \
  --itr           "${itr}" \
  --is_gpt        1 \
  --plt           0 \
  --read_model    0 \
  --write_model   0 \
  --if_inverse    0 \
  --order         1 \
  --fix_seed      2021

echo ""
echo " [1/3] Llama2 training completed"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Train PatchTST model
# ═══════════════════════════════════════════════════════════════════════════════
echo "──────────────────────────────────────────────────────"
echo " [2/3] Training PatchTST model (${TEST_DATE})"
echo "──────────────────────────────────────────────────────"

patchtst_model_id="flu_north_${patchtst_model}_pl${pred_len}_${TEST_DATE}"

python main.py \
  --root_path   "${ROOT_PATH}" \
  --data_path   "${DATA_PATH}" \
  --model_id    "${patchtst_model_id}" \
  --data        custom \
  --features    S \
  --target      positive_rate \
  --seq_len     "${seq_len}" \
  --label_len   "${label_len}" \
  --pred_len    "${pred_len}" \
  --test_end_date "${TEST_DATE}" \
  --batch_size  "${patchtst_batch_size}" \
  --learning_rate "${patchtst_lr}" \
  --train_epochs "${train_epochs}" \
  --decay_fac   0.75 \
  --d_model     "${patchtst_d_model}" \
  --n_heads     "${patchtst_n_heads}" \
  --d_ff        "${patchtst_d_ff}" \
  --freq        W \
  --patch_size  16 \
  --stride      2 \
  --percent     "${percent}" \
  --gpt_layers  "${patchtst_gpt_layer}" \
  --itr         "${itr}" \
  --model       "${patchtst_model}" \
  --is_gpt      1 \
  --plt         0 \
  --read_model  0 \
  --write_model 0

echo ""
echo " [2/3] PatchTST training completed"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Generate comparison plot
# ═══════════════════════════════════════════════════════════════════════════════
echo "──────────────────────────────────────────────────────"
echo " [3/3] Generating comparison plot"
echo "──────────────────────────────────────────────────────"

python compare_llama2_vs_patchtst.py \
  --prediction_dir ./predictions \
  --dataset_dir ./dataset \
  --figure_dir ./figures

echo ""
echo "========================================================"
echo " Comparison experiment completed!"
echo " - Llama2 predictions: ./predictions/${llama2_model_id}_itr*_predictions.csv"
echo " - PatchTST predictions: ./predictions/${patchtst_model_id}_itr*_predictions.csv"
echo " - Comparison figure: ./figures/${TEST_DATE}.png"
echo "========================================================"
