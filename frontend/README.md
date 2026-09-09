# Frontend Deployment

The frontend is static HTML/CSS/JavaScript and can be deployed to Vercel, Netlify, or GitHub Pages.

For local testing, `frontend/js/app.js` uses:

```text
http://127.0.0.1:8000/predict
```

For deployment, copy `frontend/config.example.js` to `frontend/config.js`, change the URL to your deployed backend `/predict` endpoint, and include it before `frontend/js/app.js` in `index.html`.
