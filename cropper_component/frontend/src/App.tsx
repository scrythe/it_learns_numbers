import { useEffect, useRef } from 'react';
import {
  ComponentProps,
  Streamlit,
  withStreamlitConnection,
} from 'streamlit-component-lib';
import Cropper from 'cropperjs';

function App({ args }: ComponentProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const cropperRef = useRef<Cropper>(null);
  const { image } = args;

  useEffect(() => {
    if (imageRef.current && !cropperRef.current) {
      const cropper = new Cropper(imageRef.current);
      cropperRef.current = cropper;
      const cropperCanvas = cropper.getCropperCanvas();
      const cropperImage = cropper.getCropperImage();
      cropperImage?.$ready(() => {
        Streamlit.setFrameHeight();
      });
      const cropperSelection = cropper.getCropperSelection()!;
      cropperSelection.aspectRatio = 1;
      cropperCanvas?.addEventListener('actionend', async () => {
        const croppedCanvas = await cropperSelection.$toCanvas();
        const croppedImage = croppedCanvas.toDataURL('image/png');
        console.log(croppedImage);
        Streamlit.setComponentValue(croppedImage);
      });
    }
  }, []);

  return (
    <>
      <img ref={imageRef} src={image} className="logo" alt="Vite logo" />
    </>
  );
}

// export default App
const ConnectedApp = withStreamlitConnection(App);
export default ConnectedApp;
