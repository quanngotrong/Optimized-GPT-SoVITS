## Speech-to-Text (ASR): Paraformer-Large
- Purpose: Converts spoken language (audio) into text
- Details:
    - Model: Paraformer-Large
    - Lang: chinese
    - sample rate: 16kHz, optimized for audio recorded at this sample rate
    - vocab size: 8404 tokens, representing the words/characters in the language model
- Features:
    - Non-Autoregressive Transformer (NAT): provide faster inference by predicting the entire sequence in parallel instead of step-by-step
    - Use case: speech regconition for tasks like transcription, live speech-to-text, or voice command processing
- Applications:
    - automatic transcription of meetings, lectures

## Voice activities detection (VAD): FSMD-VAD
- Purpose: identifies segments in an audio stream where speech is present and separates them from non-speech parts like silence, noise or background sounds
- Details:
    - Model: FSMN (Feedforward Sequential Memory Network)
    - Language: Chinese (zh-cn), but can generally detect speech regardless of langauge content
    - Sample Rate: 16kHz, tailored for speech input with this frequency
- Features:
    - FSMN architecture: efficient for time-series modeling, using memory blocks to capture temporal dependencies
    - Use case: detecting active speech regions in a continuous audio stream

## Punctuation Restoration: CT-Transformer
- Purpose: restores punctuation marks in transcribed text, which are tuypically absent in raw ASR
- Features:
    - Transformer Architecture: Captures contextual information to predict punctuation marks.
    - Use Case: Converts raw ASR output into well-punctuated, human-readable text.
