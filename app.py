import os
import subprocess
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv  # Import dotenv if using .env file

# Load environment variables from .env file (optional)
load_dotenv()

app = Flask(__name__)

# Cloudinary configuration from environment variables
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

# Folder untuk menyimpan input dan output
UPLOAD_FOLDER = './static/temp_input'
OUTPUT_FOLDER = './static/temp_output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Fungsi untuk memvalidasi ekstensi gambar yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_now():
    return {'current_path': request.path}

# Route utama untuk menampilkan halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menampilkan halaman prediksi upload
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Route untuk mengunggah gambar
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Jalankan perintah predict.py setelah file diupload
            command = f'python3 predict.py --input "{app.config["UPLOAD_FOLDER"]}" --output "{app.config["OUTPUT_FOLDER"]}" --width 640 --height 640 --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"'
            
            # Tambahkan logging untuk melihat command yang dijalankan
            print(f"Running command: {command}")

            process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Cek output dan error dari proses
            print(f"stdout: {process.stdout.decode()}")
            print(f"stderr: {process.stderr.decode()}")

            # Cek apakah ada error dalam eksekusi predict.py
            if process.returncode != 0:
                return f'Error running prediction: {process.stderr.decode()}', 500

            # Upload output file to Cloudinary
            output_filename = os.listdir(app.config['OUTPUT_FOLDER'])[0]  # Assuming one output file
            output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            cloudinary_response = cloudinary.uploader.upload(output_filepath)

            # Get the URL of the uploaded image
            image_url = cloudinary_response['secure_url']

            # Clean up local files (output file)
            os.remove(output_filepath)  # Remove the generated output file

            return redirect(url_for('show_result', image_url=image_url))

        finally:
            # Always clean up the uploaded input file
            if os.path.exists(filepath):
                os.remove(filepath)  # Remove the uploaded input file

    return 'File not allowed', 400

# Route untuk menampilkan hasil prediksi
@app.route('/result')
def show_result():
    image_url = request.args.get('image_url')
    return render_template('result.html', image_url=image_url)

# Route untuk mengunduh gambar
import requests
from flask import Response

# Route untuk mengunduh gambar
@app.route('/download/<filename>')
def download_file(filename):
    image_url = request.args.get('image_url')

    # Download the image from Cloudinary or any URL
    response = requests.get(image_url, stream=True)

    if response.status_code == 200:
        # Send the image as an attachment
        return Response(
            response.content,
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'image/png'  # or 'image/jpeg' based on the image type
            }
        )
    else:
        return "Failed to download the image", 500



if __name__ == "__main__":
    app.run(debug=True)
    
