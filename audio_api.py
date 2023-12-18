from flask import Flask, request, jsonify
from pydub import AudioSegment
import torchaudio
from tortoise.api import TextToSpeech
from LIHQ.procedures.tortoise_scripts import tortoise_run

app = Flask(__name__)

tts = TextToSpeech()
silence_duration = 500  # Set duration in milliseconds. 1000 is 1 second

@app.route('/generate_tts', methods=['POST'])
def generate_tts():
    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'biden')
    preset = data.get('preset', 'high_quality')

    # Generate TTS
    gen = tortoise_run(tts, text, voice, preset)

    # Save generated audio
    audio_path = 'output.wav'
    torchaudio.save(audio_path, gen.squeeze(0).cpu(), 24000)

    # Add silence before audio
    silent_segment = AudioSegment.silent(duration=silence_duration)
    final_audio = silent_segment + AudioSegment.from_file(audio_path)

    # Export the final audio
    final_audio.export(audio_path, format="wav")

    return jsonify({'audio_path': audio_path})

if __name__ == '__main__':
    # Run the app on 0.0.0.0 (accessible from any network address) and port 5001
    app.run(host='172.31.0.2', port=3000, debug=True)