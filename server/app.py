from flask import Flask, render_template, request, redirect
import os

# Create a Flask app 
app = Flask(__name__)

# upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the uploaded folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# Route for the homepage 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image was uploaded
        file = request.files.get('image')  # Get the uploaded file
        if not file:
            return "No file uploaded. Please try again.", 400
    
        # Save the uploaded file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Redirect to the results page with the filename
        return redirect(f'/result?filename={file.filename}')
    
    # for the index.html page for GET requests
    return render_template('index.html')

# Route for the results page 
@app.route('/result')
def result():
    # this gets the filename from the parameters
    filename = request.args.get('filename')

    # for the result.html page with image paths
    return render_template(
        'result.html',
        user_image=f'/static/uploads/{filename}',
        emoji_image='/static/images/emoji.png'
    )

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
