import "./App.css";
import testImage from "./Solid_red.png";
import { useState } from "react";

export default function App() {
  const [currentPage, setCurrentPage] = useState("emoji-generator");

  function Button1() {
    setCurrentPage("emoji-generator");
  }
  function Button2() {
    setCurrentPage("saved-emojis");
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>
          {currentPage === "emoji-generator"
            ? "Emoji Generator"
            : "Saved Emojis"}
        </h1>
        <div className="buttons">
          <button type="button" onClick={Button1}>
            emoji generator
          </button>
          <button type="button" onClick={Button2}>
            saved emojis
          </button>
        </div>
      </header>

      {currentPage === "emoji-generator" ? (
        <section className="App-section">
          <CameraInput />
          <EmojiOutput />
        </section>
      ) : (
        <section className="App-section">
          <SavedEmojis />
        </section>
      )}
    </div>
  );
}

export function CameraInput() {
  return (
    <div className="CameraInput-class">
      <header className="CameraInput-header">
        <img src={testImage} alt="Camera Input"></img>
        <div className="text-on-image">
          <p> Camera Input </p>
        </div>
      </header>
    </div>
  );
}

export function EmojiOutput() {
  return (
    <div className="EmojiOutput-class">
      <header className="EmojiOutput-header">
        <img src={testImage} alt="Emoji Output"></img>
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
      <img src={testImage} alt="Emoji Output"></img>
      <div className="text-on-image">
        <p> Saved Emoji </p>
      </div>
      <img src={testImage} alt="Emoji Output"></img>
      <div className="text-on-image">
        <p> Saved Emoji </p>
      </div>
      <p>To be added</p>
    </div>
  );
}
