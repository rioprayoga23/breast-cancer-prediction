import os
import subprocess
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import simplejson as json 

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
            # Prediction command
            command = f'python3 predict.py --i "{app.config["UPLOAD_FOLDER"]}" --o "{app.config["OUTPUT_FOLDER"]}" --t 0.5 --m "./saved_model" --l "./label_map.pbtxt"'
            print(f"Running command: {command}")
            process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Log command output
            print(f"stdout: {process.stdout.decode()}")
            print(f"stderr: {process.stderr.decode()}")

            # Check for prediction errors
            if process.returncode != 0:
                print(f'Error running prediction: {process.stderr.decode()}')

            # Check for output files
            output_files = os.listdir(app.config['OUTPUT_FOLDER'])
            if output_files:
                output_filename = output_files[0]  # Assuming single output file
                output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

                # Load the results from result.json
                result_json_path = './result.txt'
                try:
                    with open(result_json_path, 'r') as f:
                        result_data = json.load(f)
                        category = result_data.get("category")
                        score = result_data.get("score")
                        # Log to confirm data read from JSON
                        print(f"Loaded from result.json - Category: {category}, Score: {score}")
                except Exception as e:
                    print(f"Error reading result.json: {e}")
                    category, score = None, None

                # Upload output file to Cloudinary
                cloudinary_response = cloudinary.uploader.upload(output_filepath)
                image_url = cloudinary_response['secure_url']
                os.remove(output_filepath)
                not_predict = "F"
            else:
                print("No output detected. Uploading original image.")
                cloudinary_response = cloudinary.uploader.upload(filepath)
                image_url = cloudinary_response['secure_url']
                not_predict = "T"
                category, score = None, None  # Set to None if no prediction

            # Prepare the query parameters conditionally
            query_params = {"image_url": image_url, "np": not_predict}
            if category is not None:
                query_params["c"] = category
            if score is not None:
                query_params["s"] = score

            # Redirect to result page with conditional query parameters
            return redirect(url_for('show_result', **query_params))

        finally:
            # Clean up uploaded input file
            if os.path.exists(filepath):
                os.remove(filepath)

    return 'File not allowed', 400

# Route untuk menampilkan hasil prediksi
@app.route('/result')
def show_result():
    image_url = request.args.get('image_url')  # Ambil URL gambar jika ada
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
    
