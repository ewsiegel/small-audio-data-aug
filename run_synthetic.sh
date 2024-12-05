f5-tts_infer-cli \
 --model "E2-TTS" \
 --remove_silence True \
 --ref_audio "$1" \
 --gen_text "$2" \
 --output_dir "$3" \
 --output_file "$4"