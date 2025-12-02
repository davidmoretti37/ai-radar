# AI Radar 🛰️

Your personal AI news aggregator. One page, all the latest AI news from 16+ sources.

## Sources Included

**Labs:**
- OpenAI Blog
- Anthropic Blog  
- Google AI Blog
- DeepMind Blog

**News:**
- MIT Tech Review (AI)
- The Verge AI
- Ars Technica
- VentureBeat AI
- TechCrunch AI
- Wired AI
- The Information

**Community:**
- Hacker News (AI filtered)
- Reddit r/MachineLearning
- Reddit r/LocalLLaMA

**Tools:**
- Hugging Face Blog
- LangChain Blog

## Quick Start

1. Make sure you have Node.js installed (version 18+)
   - Download from: https://nodejs.org

2. Open terminal in this folder and run:
   ```bash
   npm install
   npm start
   ```

3. Open your browser to: http://localhost:3000

That's it! Bookmark it and check it every morning.

## Adding More Sources

Edit `server.js` and add to the `RSS_SOURCES` array:

```javascript
{ name: 'Source Name', url: 'https://example.com/rss', category: 'news' }
```

Categories: `labs`, `news`, `community`, `tools`

## Deploy Online (Optional)

To have this running 24/7 without keeping your computer on:

### Option 1: Railway (Easiest)
1. Go to railway.app
2. Connect your GitHub
3. Push this folder to a repo
4. Deploy - done

### Option 2: Render
1. Go to render.com
2. New Web Service
3. Connect repo
4. It auto-detects Node.js

### Option 3: Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in this folder
3. Follow prompts

All have free tiers that work for this.

## Customize

- Change refresh interval: Edit `setInterval` in `public/index.html` (default: 10 min)
- Change colors: Edit CSS variables at top of `public/index.html`
- Change number of articles per source: Edit `.slice(0, 15)` in `server.js`

---

Built for staying on top of AI. 🚀
