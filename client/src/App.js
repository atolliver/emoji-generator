import "./App.css";
import testImage from "./Solid_red.png";
import { useState } from "react";

export default function App() {
  const [currentPage, setCurrentPage] = useState("emoji-generator");
  const [cameraImage, setCameraImage] = useState(null);
  const [emojiImage, setEmojiImage] = useState(null);
  const [uploaded, setUploaded] = useState(false);

  function Button1() {
    setCurrentPage("emoji-generator");
  }

  function Button2() {
    setCurrentPage("saved-emojis");
  }

  function favoriteButton() {}

  function handleUpload(file) {
    if (!file) return;

    const localURL = URL.createObjectURL(file);
    setCameraImage(localURL);

    const form = new FormData();
    form.append("image", file);

    fetch("http://127.0.0.1:5000/api/upload", {
      method: "POST",
      body: form,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Uploaded:", data); // Logging
        setUploaded(true);
      })
      .catch((error) => {
        console.error("Upload failed:", error); // Logging
      });
  }

  function captureButton() {
    fetch("http://127.0.0.1:5000/api/emoji", {
      method: "GET",
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Captured:", data); // Logging
        const baseURL = "http://127.0.0.1:5000";
        setEmojiImage(baseURL + data.emoji_image);
      })
      .catch((error) => {
        console.error("Capture failed:", error); // Logging
      });
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>
          {currentPage === "emoji-generator"
            ? "Emoji Generator"
            : "Saved Emojis"}
        </h1>
        <div className="listButtons">
          <button type="button" onClick={Button1}>
            emoji generator
          </button>
          <button type="button" onClick={Button2}>
            saved emojis
          </button>
        </div>
      </header>

      {currentPage === "emoji-generator" ? (
        <>
          <section className="App-section">
            <CameraInput image={cameraImage} />
            <EmojiOutput image={emojiImage} />
          </section>
          <div className="bottomButtons">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files[0];
                handleUpload(file);
              }}
            />
            <button type="button" onClick={captureButton} disabled={!uploaded}>
              Capture
            </button>
            <button type="button" onClick={favoriteButton}>
              Favorite
            </button>
          </div>
        </>
      ) : (
        <section className="App-section">
          <SavedEmojis />
        </section>
      )}
    </div>
  );
}

export function CameraInput({ image }) {
  return (
    <div className="CameraInput-class">
      <header className="CameraInput-header">
        <img src={image || testImage} alt="Camera Input" />
        <div className="text-on-image">
          <p> Camera Input </p>
        </div>
      </header>
    </div>
  );
}

export function EmojiOutput({ image }) {
  return (
    <div className="EmojiOutput-class">
      <header className="EmojiOutput-header">
        <img src={image || testImage} alt="Emoji Output" />
        <div className="text-on-image">
          <p> Emoji Output </p>
        </div>
      </header>
    </div>
  );
}

export function SavedEmojis() {
  return (
    <div className="SavedEmojis-class">
      <img src={testImage} alt="Saved Emoji" />
      <div className="text-on-image">
        <p> Saved Emoji </p>
      </div>
      <img src={testImage} alt="Saved Emoji" />
      <div className="text-on-image">
        <p> Saved Emoji </p>
      </div>
      <p>To be added</p>
    </div>
  );
}
