from flask import Flask, render_template, request, send_file, jsonify
import webbrowser
import threading

from main import generate

app = Flask(__name__)

MIDI_PATH = "outputs/Melody_Generated.mid"


@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ''
    generated_output = ''
    error = ''

    if request.method == 'POST':
        user_input = request.form.get('music_input', '')
        note_count = int(request.form.get('note_count', 200))
        temperature = float(request.form.get('temperature', 0.8))

        try:
            generated_output = generate(
                note_count=note_count,
                temperature=temperature
            )
        except Exception as e:
            error = f"Something went wrong: {str(e)}"

    return render_template('index.html',
                           user_input=user_input,
                           generated_output=generated_output,
                           error=error)


@app.route('/download-midi')
def download_midi():
    """Send the generated MIDI file for download/playback"""
    try:
        return send_file(MIDI_PATH,
                         mimetype='audio/midi',
                         as_attachment=False,
                         download_name='Melody_Generated.mid')
    except FileNotFoundError:
        return jsonify(error="No melody generated yet. Click 'Generate Melody' first!"), 404


if __name__ == '__main__':
    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True, use_reloader=False)
