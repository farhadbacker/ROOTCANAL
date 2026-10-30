<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1n8QXhbFSpQTjHTEeTUvkUO1dAyRZwbcD

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`


apply_mean_filter: Uses cv2.blur
apply_median_filter: Uses cv2.medianBlur
apply_gaussian_filter: Uses cv2.GaussianBlur
apply_non_local_means: Uses cv2.fastNlMeansDenoising
apply_bayesian_wavelet: Uses skimage.restoration.denoise_wavelet (BayesShrink)
apply_contourlet_proxy: Uses cv2.getGaborKernel and convolution (Gabor Filter)