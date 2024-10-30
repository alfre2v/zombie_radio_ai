# zombie_radio_ai


A Python CLI Application that performs a audio-only theater play with several AI voice actors about 4 persons trapped in a lab with Zombies. The user can interact with the voice actors via voice commands at appropriate times in the story.
The crucial goal I set myself for this project is to rely only on open source AI models that anyone can run locally in a PC with a relatively modern Cuda card. No external services needed! 
AI Models involved:
1. LLM to provide dynamic text lines for each actor and to react in unexpected ways to user input (Ollama's model Nemotron-mini).
2. Text-to-Speech (TTS) model to generate emotionally charged audios for the voice lines (F5-TTS).
3. Automatic Speech Recognition (ASR) with Whisper tiny model.

The downside of this project is that it can only be run in a PC with a graphic card with at least 12GB of VRAM.  The upside is privacy and control of your own AI systems.
