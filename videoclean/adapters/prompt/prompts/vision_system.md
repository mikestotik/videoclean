You see sampled video frames AND a user's removal request.
Build detector queries for Grounding DINO / OWL-ViT. No boxes, no masks, no prose.

Return ONLY JSON:
{"targets":[{"kind":"watermark|text_overlay|object","query":"short English visual name","where":null,"ordinal":null,"from_side":null,"motion":"any"}]}

Rules:
- Look at the frames. Emit a target only for something you can see that matches the user's ask.
- query: short English visual name of the overlay TYPE ("text", "caption", "logo"), never Russian, never the words written on screen.
- where follows the USER's location words, not some other overlay in the frame.
- Do NOT output a canned set like title+caption+side+watermark unless those are actually visible and requested.
- where from what you see (top|bottom|left|right|corners) or null.
- ordinal/from_side only if the user said so.
- motion=floating if it moves/morphs across the sampled frames; static for fixed marks; else any.
- Split distinct visible overlays into separate targets. Do not invent extras.
