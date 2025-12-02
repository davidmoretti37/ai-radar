const express = require('express');
const Parser = require('rss-parser');
const path = require('path');

const app = express();
const parser = new Parser({
    timeout: 10000,
    headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
});

// Category definitions
const CATEGORIES = {
    'agents': { label: 'AI Agents', color: '#00ff88' },
    'llms': { label: 'LLMs & APIs', color: '#8b5cf6' },
    'local_ai': { label: 'Local AI', color: '#f59e0b' },
    'devtools': { label: 'Dev Tools', color: '#3b82f6' },
    'security': { label: 'Security', color: '#ef4444' },
    'learning': { label: 'Learning', color: '#10b981' },
    'research': { label: 'Research', color: '#ec4899' },
    'news': { label: 'News', color: '#6b7280' }
};

// RSS Feed sources - curated for quality
const RSS_SOURCES = [
    // === TOP-TIER COMPANY BLOGS ===
    { name: 'OpenAI Blog', url: 'https://openai.com/blog/rss.xml', category: 'llms' },
    { name: 'Google AI', url: 'https://blog.google/technology/ai/rss/', category: 'llms' },
    { name: 'Microsoft AI', url: 'https://blogs.microsoft.com/ai/feed/', category: 'llms' },

    // === AI THOUGHT LEADERS ===
    { name: 'Andrej Karpathy', url: 'https://karpathy.github.io/feed.xml', category: 'learning' },
    { name: 'Lilian Weng', url: 'https://lilianweng.github.io/index.xml', category: 'research' },
    { name: 'Jay Alammar', url: 'https://jalammar.github.io/feed.xml', category: 'learning' },
    { name: 'Chip Huyen', url: 'https://huyenchip.com/feed.xml', category: 'learning' },

    // === MORE COMPANY BLOGS ===
    { name: 'NVIDIA Blog', url: 'https://blogs.nvidia.com/feed/', category: 'devtools' },
    { name: 'AWS ML Blog', url: 'https://aws.amazon.com/blogs/machine-learning/feed/', category: 'devtools' },
    { name: 'Hugging Face', url: 'https://huggingface.co/blog/feed.xml', category: 'devtools' },
    { name: 'Together AI', url: 'https://www.together.ai/blog/rss.xml', category: 'llms' },
    { name: 'Roboflow Blog', url: 'https://blog.roboflow.com/rss/', category: 'devtools' },
    { name: 'Weaviate Blog', url: 'https://weaviate.io/blog/rss.xml', category: 'devtools' },
    { name: 'Databricks Blog', url: 'https://www.databricks.com/feed', category: 'devtools' },
    { name: 'Cloudflare AI', url: 'https://blog.cloudflare.com/tag/ai/rss/', category: 'devtools' },
    { name: 'Vespa Blog', url: 'https://blog.vespa.ai/feed.xml', category: 'devtools' },

    // === REDDIT COMMUNITIES ===
    { name: 'r/LocalLLaMA', url: 'https://www.reddit.com/r/LocalLLaMA/.rss', category: 'local_ai' },
    { name: 'r/ollama', url: 'https://www.reddit.com/r/ollama/.rss', category: 'local_ai' },
    { name: 'r/ChatGPTCoding', url: 'https://www.reddit.com/r/ChatGPTCoding/.rss', category: 'agents' },
    { name: 'r/AutoGPT', url: 'https://www.reddit.com/r/AutoGPT/.rss', category: 'agents' },
    { name: 'r/LangChain', url: 'https://www.reddit.com/r/LangChain/.rss', category: 'devtools' },
    { name: 'r/MachineLearning', url: 'https://www.reddit.com/r/MachineLearning/.rss', category: 'research' },
    { name: 'r/singularity', url: 'https://www.reddit.com/r/singularity/.rss', category: 'news' },
    { name: 'r/ClaudeAI', url: 'https://www.reddit.com/r/ClaudeAI/.rss', category: 'llms' },
    { name: 'r/OpenAI', url: 'https://www.reddit.com/r/OpenAI/.rss', category: 'llms' },
    { name: 'r/StableDiffusion', url: 'https://www.reddit.com/r/StableDiffusion/.rss', category: 'devtools' },
    { name: 'r/artificial', url: 'https://www.reddit.com/r/artificial/.rss', category: 'news' },
    { name: 'r/PromptEngineering', url: 'https://www.reddit.com/r/PromptEngineering/.rss', category: 'learning' },
    { name: 'r/Oobabooga', url: 'https://www.reddit.com/r/Oobabooga/.rss', category: 'local_ai' },
    { name: 'r/ComfyUI', url: 'https://www.reddit.com/r/ComfyUI/.rss', category: 'devtools' },
    { name: 'r/mlops', url: 'https://www.reddit.com/r/mlops/.rss', category: 'devtools' },

    // === GITHUB RELEASES ===
    { name: 'LangChain Releases', url: 'https://github.com/langchain-ai/langchain/releases.atom', category: 'devtools', isRelease: true },
    { name: 'CrewAI Releases', url: 'https://github.com/crewAIInc/crewAI/releases.atom', category: 'agents', isRelease: true },
    { name: 'AutoGen Releases', url: 'https://github.com/microsoft/autogen/releases.atom', category: 'agents', isRelease: true },
    { name: 'Ollama Releases', url: 'https://github.com/ollama/ollama/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'llama.cpp Releases', url: 'https://github.com/ggerganov/llama.cpp/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Chroma Releases', url: 'https://github.com/chroma-core/chroma/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Open Interpreter', url: 'https://github.com/OpenInterpreter/open-interpreter/releases.atom', category: 'agents', isRelease: true },
    { name: 'Cursor Releases', url: 'https://github.com/getcursor/cursor/releases.atom', category: 'devtools', isRelease: true },
    { name: 'vLLM Releases', url: 'https://github.com/vllm-project/vllm/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Transformers', url: 'https://github.com/huggingface/transformers/releases.atom', category: 'devtools', isRelease: true },
    { name: 'LlamaIndex Releases', url: 'https://github.com/run-llama/llama_index/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Dify Releases', url: 'https://github.com/langgenius/dify/releases.atom', category: 'agents', isRelease: true },
    { name: 'Flowise Releases', url: 'https://github.com/FlowiseAI/Flowise/releases.atom', category: 'agents', isRelease: true },
    { name: 'GPT4All Releases', url: 'https://github.com/nomic-ai/gpt4all/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'LocalAI Releases', url: 'https://github.com/mudler/LocalAI/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Text Gen WebUI', url: 'https://github.com/oobabooga/text-generation-webui/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'ComfyUI Releases', url: 'https://github.com/comfyanonymous/ComfyUI/releases.atom', category: 'devtools', isRelease: true },
    { name: 'SD WebUI Releases', url: 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases.atom', category: 'devtools', isRelease: true },

    // === NEWSLETTERS & SUBSTACKS ===
    { name: 'Latent Space', url: 'https://www.latent.space/feed', category: 'learning' },
    { name: 'AI Snake Oil', url: 'https://aisnakeoil.substack.com/feed', category: 'news' },
    { name: 'Ahead of AI', url: 'https://magazine.sebastianraschka.com/feed', category: 'learning' },
    { name: 'AIModels.fyi', url: 'https://aimodels.substack.com/feed', category: 'research' },
    { name: 'The Gradient', url: 'https://thegradient.pub/rss/', category: 'research' },
    { name: 'Import AI', url: 'https://importai.substack.com/feed', category: 'news' },

    // === LEARNING (focused on LLMs/AI) ===
    { name: 'Simon Willison', url: 'https://simonwillison.net/atom/everything/', category: 'learning' },

    // === RESEARCH ===
    { name: 'arXiv cs.AI', url: 'https://rss.arxiv.org/rss/cs.AI', category: 'research' },
    { name: 'arXiv cs.CL', url: 'https://rss.arxiv.org/rss/cs.CL', category: 'research' },

    // === CURATED AI NEWS (quality over quantity) ===
    { name: 'Last Week in AI', url: 'https://lastweekin.ai/feed', category: 'news' },
    { name: 'Hacker News AI', url: 'https://hnrss.org/newest?q=LLM+OR+GPT+OR+Claude+OR+AI+agent&points=100', category: 'news' },
    { name: 'IEEE Spectrum AI', url: 'https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss', category: 'research' },

    // === MORE YOUTUBE CHANNELS ===
    { name: 'Two Minute Papers', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg', category: 'learning' },
    { name: 'Yannic Kilcher', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCZHmQk67mSJgfCCTn7xBfew', category: 'research' },
    { name: 'AI Explained', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCNJ1Ymd5yFuUPtn21xtRbbw', category: 'learning' },
    { name: 'Matt Wolfe', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCJIfeSCssxSC_Dhc5s7woww', category: 'news' },
    { name: 'AI Jason', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCfAkVzpSvSX1mVFVK9JZ9WA', category: 'learning' },
    { name: 'Sam Witteveen', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCTclNJdFSBrJgXJEemQu_dA', category: 'learning' },
    { name: 'James Briggs', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCv83tO5cePwHMt1952IVVHw', category: 'learning' },
    { name: 'AI Advantage', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCBNHHEoiSF8pcLgqLKVugOw', category: 'learning' },
    { name: 'Matthew Berman', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCRI-Ds5eY70wSdkLBNkhHvg', category: 'local_ai' },
    { name: 'All About AI', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UC4knBXsHW7dHz-qVjIqGKow', category: 'learning' },
    { name: 'WorldofAI', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCylGUf9BvQooEFjgdNudoQg', category: 'news' },
    { name: 'bycloud', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCgfe2ooZD3VJPB6aJAnuQng', category: 'learning' },

    // === MORE GITHUB RELEASES ===
    { name: 'Whisper Releases', url: 'https://github.com/openai/whisper/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Langflow Releases', url: 'https://github.com/langflow-ai/langflow/releases.atom', category: 'agents', isRelease: true },
    { name: 'MemGPT Releases', url: 'https://github.com/cpacker/MemGPT/releases.atom', category: 'agents', isRelease: true },
    { name: 'Continue Releases', url: 'https://github.com/continuedev/continue/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Aider Releases', url: 'https://github.com/paul-gauthier/aider/releases.atom', category: 'devtools', isRelease: true },
    { name: 'OpenDevin Releases', url: 'https://github.com/OpenDevin/OpenDevin/releases.atom', category: 'agents', isRelease: true },
    { name: 'Jan Releases', url: 'https://github.com/janhq/jan/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Anything LLM', url: 'https://github.com/Mintplex-Labs/anything-llm/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Koboldcpp Releases', url: 'https://github.com/LostRuins/koboldcpp/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Tabby Releases', url: 'https://github.com/TabbyML/tabby/releases.atom', category: 'devtools', isRelease: true },
    { name: 'FastChat Releases', url: 'https://github.com/lm-sys/FastChat/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'ExLlamaV2 Releases', url: 'https://github.com/turboderp/exllamav2/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'MLX Releases', url: 'https://github.com/ml-explore/mlx/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Phidata Releases', url: 'https://github.com/phidatahq/phidata/releases.atom', category: 'agents', isRelease: true },
    { name: 'Semantic Kernel', url: 'https://github.com/microsoft/semantic-kernel/releases.atom', category: 'devtools', isRelease: true },
    { name: 'LiteLLM Releases', url: 'https://github.com/BerriAI/litellm/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Instructor Releases', url: 'https://github.com/jxnl/instructor/releases.atom', category: 'devtools', isRelease: true },
    { name: 'DSPy Releases', url: 'https://github.com/stanfordnlp/dspy/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Guidance Releases', url: 'https://github.com/guidance-ai/guidance/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Outlines Releases', url: 'https://github.com/outlines-dev/outlines/releases.atom', category: 'devtools', isRelease: true },

    // === MORE NEWSLETTERS/SUBSTACKS ===
    { name: 'The Neuron', url: 'https://www.theneurondaily.com/feed', category: 'news' },
    { name: "Ben's Bites", url: 'https://bensbites.beehiiv.com/feed', category: 'news' },
    { name: 'Superhuman AI', url: 'https://www.superhuman.ai/feed', category: 'news' },
    { name: 'The Rundown AI', url: 'https://www.therundown.ai/feed', category: 'news' },
    { name: 'AI Supremacy', url: 'https://aisupremacy.substack.com/feed', category: 'news' },
    { name: 'The Algorithm', url: 'https://the-algorithm.beehiiv.com/feed', category: 'news' },
    { name: 'Davis Summarizes', url: 'https://davissummarizes.substack.com/feed', category: 'learning' },
    { name: 'Interconnects', url: 'https://www.interconnects.ai/feed', category: 'research' },
    { name: 'Lil Log', url: 'https://lilianweng.github.io/index.xml', category: 'research' },

    // === MORE SUBREDDITS ===
    { name: 'r/learnmachinelearning', url: 'https://www.reddit.com/r/learnmachinelearning/.rss', category: 'learning' },
    { name: 'r/ArtificialIntelligence', url: 'https://www.reddit.com/r/ArtificialIntelligence/.rss', category: 'news' },
    { name: 'r/GPT3', url: 'https://www.reddit.com/r/GPT3/.rss', category: 'llms' },
    { name: 'r/Bard', url: 'https://www.reddit.com/r/Bard/.rss', category: 'llms' },
    { name: 'r/midjourney', url: 'https://www.reddit.com/r/midjourney/.rss', category: 'devtools' },
    { name: 'r/deeplearning', url: 'https://www.reddit.com/r/deeplearning/.rss', category: 'research' },
    { name: 'r/MLQuestions', url: 'https://www.reddit.com/r/MLQuestions/.rss', category: 'learning' },
    { name: 'r/datascience', url: 'https://www.reddit.com/r/datascience/.rss', category: 'learning' },
    { name: 'r/agi', url: 'https://www.reddit.com/r/agi/.rss', category: 'research' },
    { name: 'r/aivideo', url: 'https://www.reddit.com/r/aivideo/.rss', category: 'devtools' },

    // === MORE COMPANY BLOGS ===
    { name: 'Groq Blog', url: 'https://groq.com/feed/', category: 'llms' },
    { name: 'Anyscale Blog', url: 'https://www.anyscale.com/blog/rss.xml', category: 'devtools' },
    { name: 'Modal Blog', url: 'https://modal.com/blog/feed.xml', category: 'devtools' },
    { name: 'W&B Blog', url: 'https://wandb.ai/site/feed.xml', category: 'devtools' },
    { name: 'Pinecone Blog', url: 'https://www.pinecone.io/blog/rss.xml', category: 'devtools' },
    { name: 'Qdrant Blog', url: 'https://qdrant.tech/blog/rss.xml', category: 'devtools' },

    // === PODCASTS ===
    { name: 'Latent Space Pod', url: 'https://api.substack.com/feed/podcast/1084089.rss', category: 'learning' },
    { name: 'Practical AI', url: 'https://changelog.com/practicalai/feed', category: 'learning' },
    { name: 'TWIML AI', url: 'https://twimlai.com/feed/', category: 'learning' },
    { name: 'Gradient Dissent', url: 'https://feeds.soundcloud.com/users/soundcloud:users:736386295/sounds.rss', category: 'learning' },
    { name: 'Eye on AI', url: 'https://www.eye-on.ai/podcast-rss.xml', category: 'news' },

    // === MORE RESEARCH ===
    { name: 'arXiv cs.LG', url: 'https://rss.arxiv.org/rss/cs.LG', category: 'research' },
    { name: 'arXiv cs.CV', url: 'https://rss.arxiv.org/rss/cs.CV', category: 'research' },
    { name: 'arXiv cs.NE', url: 'https://rss.arxiv.org/rss/cs.NE', category: 'research' },
    { name: 'arXiv cs.RO', url: 'https://rss.arxiv.org/rss/cs.RO', category: 'research' },
    { name: 'arXiv stat.ML', url: 'https://rss.arxiv.org/rss/stat.ML', category: 'research' },
    { name: 'Distill.pub', url: 'https://distill.pub/rss.xml', category: 'research' },
    { name: 'Google Research', url: 'https://blog.research.google/feeds/posts/default?alt=rss', category: 'research' },
    { name: 'Berkeley AI', url: 'https://bair.berkeley.edu/blog/feed.xml', category: 'research' },
    { name: 'CMU ML Blog', url: 'https://blog.ml.cmu.edu/feed/', category: 'research' },
    { name: 'Allen AI Blog', url: 'https://blog.allenai.org/feed', category: 'research' },

    // === EVEN MORE YOUTUBE CHANNELS ===
    { name: 'Sentdex', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCfzlCWGWYyIQ0aLC5w48gBQ', category: 'learning' },
    { name: 'Nicholas Renotte', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCHXa4OpASJEwrHrLeIzw7Yg', category: 'learning' },
    { name: 'AssemblyAI', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCtatfZMf-8EkIwASXM4ts0A', category: 'learning' },
    { name: 'Fireship', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCsBjURrPoezykLs9EqgamOA', category: 'learning' },
    { name: 'Code to the Moon', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCPHBb8D4DrTBSNYAggrUh1A', category: 'learning' },
    { name: 'Dave Ebbelaar', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCwpuMChDmlxxrEqtB7gqwrg', category: 'learning' },
    { name: 'AI Foundations', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCgBncpylJ1kiVaPyP-PZauQ', category: 'learning' },
    { name: 'Prompt Engineering', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCDq7SjPgQn75J-LPFAuOLiw', category: 'learning' },
    { name: '1littlecoder', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCpY2NKsKC2U-zaTBs3xHLcQ', category: 'learning' },
    { name: 'AI Andy', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCILl3mkpPVtyA6v4rEJUfgA', category: 'learning' },
    { name: 'AI Search', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UC4JX40jDee_tINbkjycV4Sg', category: 'learning' },
    { name: 'TheAIGRID', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCJHv2pv9UpEBALLYTb2Lhlg', category: 'news' },
    { name: 'AI Revolution', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UCOqaHzCpqvQbi7qdMCRIk4A', category: 'news' },
    { name: 'Wes Roth', url: 'https://www.youtube.com/feeds/videos.xml?channel_id=UC7z7MXZ_SvmO5XFVFR2_4Eg', category: 'news' },

    // === EVEN MORE GITHUB RELEASES ===
    { name: 'OpenRouter', url: 'https://github.com/OpenRouterTeam/openrouter-runner/releases.atom', category: 'devtools', isRelease: true },
    { name: 'txtai Releases', url: 'https://github.com/neuml/txtai/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Haystack Releases', url: 'https://github.com/deepset-ai/haystack/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Unsloth Releases', url: 'https://github.com/unslothai/unsloth/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'LMStudio', url: 'https://github.com/lmstudio-ai/lms/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Privateai', url: 'https://github.com/privategpt/privategpt/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'h2oGPT Releases', url: 'https://github.com/h2oai/h2ogpt/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'RWKV Releases', url: 'https://github.com/BlinkDL/RWKV-LM/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Mojo Releases', url: 'https://github.com/modularml/mojo/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Triton Releases', url: 'https://github.com/openai/triton/releases.atom', category: 'devtools', isRelease: true },
    { name: 'TensorRT-LLM', url: 'https://github.com/NVIDIA/TensorRT-LLM/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Candle Releases', url: 'https://github.com/huggingface/candle/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Axolotl Releases', url: 'https://github.com/OpenAccess-AI-Collective/axolotl/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'Mergekit Releases', url: 'https://github.com/cg123/mergekit/releases.atom', category: 'local_ai', isRelease: true },
    { name: 'LangGraph Releases', url: 'https://github.com/langchain-ai/langgraph/releases.atom', category: 'agents', isRelease: true },
    { name: 'Pydantic AI', url: 'https://github.com/pydantic/pydantic-ai/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Marvin Releases', url: 'https://github.com/PrefectHQ/marvin/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Chainlit Releases', url: 'https://github.com/Chainlit/chainlit/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Streamlit Releases', url: 'https://github.com/streamlit/streamlit/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Gradio Releases', url: 'https://github.com/gradio-app/gradio/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Panel Releases', url: 'https://github.com/holoviz/panel/releases.atom', category: 'devtools', isRelease: true },
    { name: 'BentoML Releases', url: 'https://github.com/bentoml/BentoML/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Ray Releases', url: 'https://github.com/ray-project/ray/releases.atom', category: 'devtools', isRelease: true },
    { name: 'MLflow Releases', url: 'https://github.com/mlflow/mlflow/releases.atom', category: 'devtools', isRelease: true },
    { name: 'DVC Releases', url: 'https://github.com/iterative/dvc/releases.atom', category: 'devtools', isRelease: true },
    { name: 'PEFT Releases', url: 'https://github.com/huggingface/peft/releases.atom', category: 'devtools', isRelease: true },
    { name: 'TRL Releases', url: 'https://github.com/huggingface/trl/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Datasets Releases', url: 'https://github.com/huggingface/datasets/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Accelerate Releases', url: 'https://github.com/huggingface/accelerate/releases.atom', category: 'devtools', isRelease: true },
    { name: 'DeepSpeed Releases', url: 'https://github.com/microsoft/DeepSpeed/releases.atom', category: 'devtools', isRelease: true },
    { name: 'FSDP PyTorch', url: 'https://github.com/pytorch/pytorch/releases.atom', category: 'devtools', isRelease: true },
    { name: 'JAX Releases', url: 'https://github.com/google/jax/releases.atom', category: 'devtools', isRelease: true },
    { name: 'Flax Releases', url: 'https://github.com/google/flax/releases.atom', category: 'devtools', isRelease: true },
    { name: 'SWE-agent', url: 'https://github.com/princeton-nlp/SWE-agent/releases.atom', category: 'agents', isRelease: true },
    { name: 'GPT Engineer', url: 'https://github.com/gpt-engineer-org/gpt-engineer/releases.atom', category: 'agents', isRelease: true },
    { name: 'Devika Releases', url: 'https://github.com/stitionai/devika/releases.atom', category: 'agents', isRelease: true },
    { name: 'AgentGPT Releases', url: 'https://github.com/reworkd/AgentGPT/releases.atom', category: 'agents', isRelease: true },
    { name: 'SuperAGI Releases', url: 'https://github.com/TransformerOptimus/SuperAGI/releases.atom', category: 'agents', isRelease: true },
    { name: 'BabyAGI Releases', url: 'https://github.com/yoheinakajima/babyagi/releases.atom', category: 'agents', isRelease: true },
    { name: 'MetaGPT Releases', url: 'https://github.com/geekan/MetaGPT/releases.atom', category: 'agents', isRelease: true },
    { name: 'ChatDev Releases', url: 'https://github.com/OpenBMB/ChatDev/releases.atom', category: 'agents', isRelease: true },

    // === MORE SUBREDDITS ===
    { name: 'r/LLMDevs', url: 'https://www.reddit.com/r/LLMDevs/.rss', category: 'devtools' },
    { name: 'r/ChatGPT', url: 'https://www.reddit.com/r/ChatGPT/.rss', category: 'llms' },
    { name: 'r/Anthropic', url: 'https://www.reddit.com/r/Anthropic/.rss', category: 'llms' },
    { name: 'r/perplexity_ai', url: 'https://www.reddit.com/r/perplexity_ai/.rss', category: 'llms' },
    { name: 'r/SillyTavernAI', url: 'https://www.reddit.com/r/SillyTavernAI/.rss', category: 'local_ai' },
    { name: 'r/KoboldAI', url: 'https://www.reddit.com/r/KoboldAI/.rss', category: 'local_ai' },
    { name: 'r/Bing', url: 'https://www.reddit.com/r/bing/.rss', category: 'llms' },
    { name: 'r/GoogleGeminiAI', url: 'https://www.reddit.com/r/GoogleGeminiAI/.rss', category: 'llms' },
    { name: 'r/reinforcementlearning', url: 'https://www.reddit.com/r/reinforcementlearning/.rss', category: 'research' },
    { name: 'r/computervision', url: 'https://www.reddit.com/r/computervision/.rss', category: 'research' },
    { name: 'r/NLP', url: 'https://www.reddit.com/r/LanguageTechnology/.rss', category: 'research' },
    { name: 'r/AIethics', url: 'https://www.reddit.com/r/AIethics/.rss', category: 'news' },
    { name: 'r/ControlProblem', url: 'https://www.reddit.com/r/ControlProblem/.rss', category: 'security' },
    { name: 'r/aiwars', url: 'https://www.reddit.com/r/aiwars/.rss', category: 'news' },
    { name: 'r/DefendingAIArt', url: 'https://www.reddit.com/r/DefendingAIArt/.rss', category: 'devtools' },
    { name: 'r/AIAssisted', url: 'https://www.reddit.com/r/AIAssisted/.rss', category: 'devtools' },
    { name: 'r/generativeAI', url: 'https://www.reddit.com/r/generativeAI/.rss', category: 'news' },

    // === MORE NEWSLETTERS ===
    { name: 'AI Weekly', url: 'https://aiweekly.co/feed/', category: 'news' },
    { name: 'Deep Learning Weekly', url: 'https://www.deeplearningweekly.com/feed', category: 'learning' },
    { name: 'NLP News', url: 'http://nlpnews.substack.com/feed', category: 'research' },
    { name: 'The Sequence', url: 'https://thesequence.substack.com/feed', category: 'learning' },
    { name: 'Davis Blalock', url: 'https://dblalock.substack.com/feed', category: 'research' },
    { name: 'Nathan Lambert', url: 'https://robotic.substack.com/feed', category: 'research' },
    { name: 'One Useful Thing', url: 'https://www.oneusefulthing.org/feed', category: 'learning' },
    { name: 'Stratechery', url: 'https://stratechery.com/feed/', category: 'news' },
    { name: 'No Priors Pod', url: 'https://www.nopriors.ai/feed', category: 'news' },

    // === VECTOR DB & DATA BLOGS ===
    { name: 'Snowflake AI', url: 'https://www.snowflake.com/blog/feed/', category: 'devtools' },
    { name: 'Vercel AI', url: 'https://vercel.com/blog/rss.xml', category: 'devtools' },
    { name: 'Milvus Blog', url: 'https://milvus.io/blog/rss.xml', category: 'devtools' },
    { name: 'Zilliz Blog', url: 'https://zilliz.com/blog/rss.xml', category: 'devtools' },

    // === AI SAFETY (focused) ===
    { name: 'EleutherAI Blog', url: 'https://blog.eleuther.ai/rss/', category: 'research' },

    // === MORE PODCASTS ===
    { name: 'Lex Fridman AI', url: 'https://lexfridman.com/feed/podcast/', category: 'learning' },
    { name: 'Data Skeptic', url: 'https://dataskeptic.libsyn.com/rss', category: 'learning' },
    { name: 'AI in Business', url: 'https://emerj.com/feed/', category: 'news' },
    { name: 'The AI Podcast', url: 'https://feeds.soundcloud.com/users/soundcloud:users:244949847/sounds.rss', category: 'learning' },
];

// Keywords for content classification
const CONTENT_KEYWORDS = {
    agents: ['agent', 'autonomous', 'crewai', 'autogen', 'workflow', 'automation', 'multi-agent', 'agentic', 'langraph'],
    llms: ['gpt-4', 'gpt-5', 'claude', 'gemini', 'api', 'openai', 'anthropic', 'tokens', 'context window', 'o1', 'o3'],
    local_ai: ['ollama', 'llama', 'mistral', 'local', 'self-hosted', 'open-source', 'gguf', 'quantization', 'mlx'],
    devtools: ['langchain', 'vector', 'embedding', 'rag', 'prompt', 'framework', 'sdk', 'library', 'chroma', 'pinecone'],
    security: ['injection', 'jailbreak', 'adversarial', 'red team', 'safety', 'alignment', 'vulnerability', 'attack'],
    learning: ['tutorial', 'guide', 'how to', 'learn', 'course', 'beginner', 'introduction', 'explained'],
    releases: ['release', 'version', 'update', 'launch', 'announcement', 'new model', 'available now', 'introducing']
};

// Keywords to filter out off-topic content
const FLUFF_KEYWORDS = [
    // Clickbait
    'celebrity', 'kicked out', 'shocking', 'you wont believe', 'clickbait',
    'relationship', 'dating', 'viral video', 'tiktok', 'influencer drama',
    'worst thing', 'heartbreaking', 'unbelievable', 'mind-blowing news',
    // Off-topic science/misc
    'planet', 'astronomy', 'space station', 'nasa moon', 'asteroid',
    'climate change', 'quantum physics', 'black hole', 'mars rover',
    // Generic non-AI
    'cryptocurrency', 'bitcoin price', 'stock market', 'forex',
    'weight loss', 'diet tips', 'fitness routine', 'health advice',
    // Low-quality Reddit patterns
    'eli5', 'ama request', 'unpopular opinion'
];

// High-authority sources get score boosts
const AUTHORITY_SCORES = {
    // Top-tier companies
    'OpenAI Blog': 35,
    'Google AI': 30,
    'Microsoft AI': 25,
    // Thought leaders (high signal)
    'Andrej Karpathy': 35,
    'Lilian Weng': 30,
    'Jay Alammar': 28,
    'Chip Huyen': 28,
    'Simon Willison': 25,
    // Quality sources
    'Hugging Face': 22,
    'LangChain Blog': 20,
    'Latent Space': 22,
    'Ahead of AI': 22,
    'Import AI': 20,
    // Curated newsletters
    'Last Week in AI': 30,
};

// Calculate highlight score for an article
function calculateHighlightScore(article) {
    let score = 0;
    const title = (article.title || '').toLowerCase();
    const description = (article.description || '').toLowerCase();
    const content = title + ' ' + description;

    // Source authority
    score += AUTHORITY_SCORES[article.source] || 5;

    // Release/announcement detection
    if (/release|launch|announc|available now|introducing/i.test(content)) score += 25;
    if (/gpt-5|gpt-4o|claude\s*4|claude\s*3\.5|gemini\s*2|o1|o3|new model/i.test(content)) score += 30;

    // Technical depth indicators
    if (/benchmark|performance|accuracy|eval|comparison/i.test(content)) score += 10;
    if (/code|github|implementation|tutorial|guide/i.test(content)) score += 15;

    // Agent-related content (user's main interest)
    if (/agent|crewai|autogen|langraph|autonomous/i.test(content)) score += 20;

    // Local AI content
    if (/ollama|llama\.cpp|local|self-host|gguf/i.test(content)) score += 15;

    // Recency bonus
    const hoursAgo = (Date.now() - new Date(article.pubDate)) / 3600000;
    if (hoursAgo < 6) score += 20;
    else if (hoursAgo < 24) score += 10;

    // Penalize fluff
    if (FLUFF_KEYWORDS.some(kw => content.includes(kw.toLowerCase()))) score -= 50;

    return score;
}

// Detect if article is about a release
function detectRelease(article) {
    const content = (article.title || '') + ' ' + (article.description || '');
    const patterns = [
        /\b(v?\d+\.\d+(\.\d+)?)\b/,
        /release|launch|now available|introducing|announcing/i,
        /new (version|release|model|api|feature)/i
    ];

    const isRelease = patterns.some(p => p.test(content)) || article.sourceIsRelease;
    const versionMatch = content.match(/\b(v?\d+\.\d+(\.\d+)?)\b/);

    return {
        isRelease,
        version: versionMatch ? versionMatch[0] : null
    };
}

// Check if article is fluff/clickbait
function isFluff(article) {
    const content = ((article.title || '') + ' ' + (article.description || '')).toLowerCase();
    return FLUFF_KEYWORDS.some(kw => content.includes(kw.toLowerCase()));
}

// Detect if article is a tutorial/learning content
function isTutorial(article) {
    const content = ((article.title || '') + ' ' + (article.description || '')).toLowerCase();
    return CONTENT_KEYWORDS.learning.some(kw => content.includes(kw));
}

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// API endpoint to fetch all feeds
app.get('/api/feeds', async (req, res) => {
    const { search } = req.query;
    console.log('Fetching all feeds...' + (search ? ` (search: "${search}")` : ''));

    const results = await Promise.all(
        RSS_SOURCES.map(async (source) => {
            try {
                const feed = await parser.parseURL(source.url);
                // Limit arXiv to 10 items, others to 12
                const limit = source.url.includes('arxiv') ? 10 : 12;
                const articles = feed.items.slice(0, limit).map(item => ({
                    title: item.title || '',
                    link: item.link || '',
                    description: (item.contentSnippet || item.content || item.summary || '').replace(/<[^>]*>/g, '').substring(0, 300),
                    pubDate: item.pubDate || item.updated ? new Date(item.pubDate || item.updated).toISOString() : new Date().toISOString(),
                    source: source.name,
                    category: source.category,
                    sourceIsRelease: source.isRelease || false
                }));
                console.log(`✓ ${source.name}: ${articles.length} articles`);
                return articles;
            } catch (error) {
                console.log(`✗ ${source.name}: ${error.message}`);
                return [];
            }
        })
    );

    // Flatten and process articles
    let allArticles = results.flat();

    // Filter out fluff
    allArticles = allArticles.filter(a => !isFluff(a));

    // Search filter
    if (search && search.trim()) {
        const searchLower = search.toLowerCase();
        allArticles = allArticles.filter(article => {
            const title = (article.title || '').toLowerCase();
            const description = (article.description || '').toLowerCase();
            const source = (article.source || '').toLowerCase();
            return title.includes(searchLower) ||
                   description.includes(searchLower) ||
                   source.includes(searchLower);
        });
    }

    // Add scores and metadata
    allArticles = allArticles.map(article => {
        const releaseInfo = detectRelease(article);
        return {
            ...article,
            highlightScore: calculateHighlightScore(article),
            isRelease: releaseInfo.isRelease,
            version: releaseInfo.version,
            isTutorial: isTutorial(article)
        };
    });

    // Sort by date
    allArticles.sort((a, b) => new Date(b.pubDate) - new Date(a.pubDate));

    // Extract special sections
    const highlights = [...allArticles]
        .sort((a, b) => b.highlightScore - a.highlightScore)
        .slice(0, 10);

    const releases = allArticles
        .filter(a => a.isRelease)
        .slice(0, 10);

    const tutorials = allArticles
        .filter(a => a.isTutorial)
        .slice(0, 10);

    console.log(`Total: ${allArticles.length} articles from ${RSS_SOURCES.length} sources`);

    res.json({
        articles: allArticles,
        highlights,
        releases,
        tutorials,
        categories: CATEGORIES,
        sources: RSS_SOURCES.length,
        updated: new Date().toISOString()
    });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`
╔═══════════════════════════════════════════╗
║        AI RADAR - Developer Edition       ║
║                                           ║
║   Open: http://localhost:${PORT}             ║
║                                           ║
║   Sources: ${RSS_SOURCES.length} feeds                       ║
╚═══════════════════════════════════════════╝
    `);
});
