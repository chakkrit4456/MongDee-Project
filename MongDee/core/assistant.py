"""AI Product Assistant: speaks product info and answers visitor questions.

TTS runs on a single dedicated background thread with a queue — pyttsx3's
engine is not safe to drive with overlapping runAndWait() calls, and
runAndWait() blocks, so it must never run on the GUI thread.
"""

from __future__ import annotations

import queue
import threading

from core.products import ProductCatalog


class AIAssistant:
    def __init__(self, catalog: ProductCatalog):
        self.catalog = catalog
        self.tts_available = False
        self.tts_error: str | None = None
        self._engine = None
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None

        self._init_tts()

    def _init_tts(self):
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self.tts_available = True
            self._worker = threading.Thread(target=self._tts_loop, daemon=True)
            self._worker.start()
        except Exception as exc:  # pyttsx3 needs a native speech driver (e.g. espeak-ng)
            self._engine = None
            self.tts_available = False
            self.tts_error = str(exc)

    def _tts_loop(self):
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                self.tts_available = False
                self.tts_error = str(exc)
                return

    def speak(self, text: str) -> None:
        if self.tts_available:
            self._queue.put(text)

    def describe(self, class_name: str, speak: bool = True) -> dict | None:
        product = self.catalog.get(class_name)
        if not product:
            return None
        if speak:
            self.speak(f"{product['name']}. {product['description']}")
        return product

    def answer(self, class_name: str, question: str, speak: bool = True) -> str:
        answer_text = self.catalog.answer_question(class_name, question)
        if speak:
            self.speak(answer_text)
        return answer_text
