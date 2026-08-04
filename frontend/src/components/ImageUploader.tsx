import { ChangeEvent, DragEvent, useId, useState } from 'react';
import type { SelectedImage } from '../types/analysis';

type ImageUploaderProps = {
  selectedImage: SelectedImage | null;
  onImageSelected: (file: File) => void;
};

const acceptedImageTypes = ['image/jpeg', 'image/png', 'image/webp'];

export function ImageUploader({ selectedImage, onImageSelected }: ImageUploaderProps) {
  const inputId = useId();
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  function handleFile(file: File | undefined) {
    if (!file) {
      return;
    }

    if (!acceptedImageTypes.includes(file.type)) {
      setError('Choose a JPEG, PNG, or WebP image.');
      return;
    }

    setError('');
    onImageSelected(file);
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    handleFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  }

  return (
    <section className="panel upload-panel" aria-labelledby="upload-title">
      <div className="section-heading">
        <p className="eyebrow">Room image</p>
        <h2 id="upload-title">Upload a room photo</h2>
      </div>

      <label
        className={`upload-zone${isDragging ? ' is-dragging' : ''}`}
        htmlFor={inputId}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <span className="upload-icon" aria-hidden="true">
          +
        </span>
        <span className="upload-title">Select or drop an image</span>
        <span className="upload-copy">JPEG, PNG, and WebP files are supported.</span>
        <input
          id={inputId}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleInputChange}
        />
      </label>

      {error ? <p className="message message-error">{error}</p> : null}

      {selectedImage ? (
        <div className="preview">
          <img src={selectedImage.previewUrl} alt="Selected room preview" />
          <div>
            <p className="preview-name">{selectedImage.file.name}</p>
            <p className="preview-meta">
              {(selectedImage.file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        </div>
      ) : null}
    </section>
  );
}
