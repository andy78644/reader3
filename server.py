import os
import pickle
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import google.generativeai as genai

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "."

@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    # Scan directory for folders ending in '_data' that have a book.pkl
    if os.path.exists(BOOKS_DIR):
        for item in os.listdir(BOOKS_DIR):
            if item.endswith("_data") and os.path.isdir(item):
                # Try to load it to get the title
                book = load_book_cached(item)
                if book:
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine)
                    })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def redirect_to_first_chapter(book_id: str):
    """Helper to just go to chapter 0."""
    return await read_chapter(book_id=book_id, chapter_index=0)

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_index: int):
    """The main reader interface."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    current_chapter = book.spine[chapter_index]

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx
    })

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)


# ========== AI CHAT API ==========

class ChatRequest(BaseModel):
    message: str
    book_id: str
    chapter_index: int
    api_key: str
    selected_text: Optional[str] = None  # Optional selected text for discussion


@app.post("/api/chat")
async def chat_with_ai(chat_request: ChatRequest):
    """
    Handle AI chat requests using Gemini API.
    """
    try:
        # Load the book
        book = load_book_cached(chat_request.book_id)
        if not book:
            return JSONResponse(
                status_code=404,
                content={"error": "書籍未找到"}
            )

        # Get current chapter
        if chat_request.chapter_index < 0 or chat_request.chapter_index >= len(book.spine):
            return JSONResponse(
                status_code=404,
                content={"error": "章節未找到"}
            )

        current_chapter = book.spine[chat_request.chapter_index]

        # Configure Gemini API
        genai.configure(api_key=chat_request.api_key)

        # Create the model (use the correct model name)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Prepare the context
        # If user selected specific text, prioritize that
        if chat_request.selected_text:
            book_context = f"""
書籍資訊：
- 標題：{book.metadata.title}
- 作者：{', '.join(book.metadata.authors) if book.metadata.authors else '未知'}
- 當前章節：{chat_request.chapter_index + 1} / {len(book.spine)}

用戶選取的文字：
「{chat_request.selected_text}」

周圍上下文（當前章節部分內容）：
{current_chapter.text[:2000]}
"""
        else:
            book_context = f"""
書籍資訊：
- 標題：{book.metadata.title}
- 作者：{', '.join(book.metadata.authors) if book.metadata.authors else '未知'}
- 語言：{book.metadata.language or '未知'}
- 當前章節：{chat_request.chapter_index + 1} / {len(book.spine)}

當前章節內容：
{current_chapter.text[:3000]}  # 限制在 3000 字元以避免超過 token 限制
"""

        # Create the prompt
        if chat_request.selected_text:
            system_prompt = """你是一個專業的閱讀助手，專門幫助讀者理解和討論書籍內容。
你的任務是：
1. 針對用戶選取的特定文字提供深入分析
2. 解釋文字的意義、背景或重要性
3. 如果適用，可以聯繫整個章節或書籍的脈絡
4. 用清晰、友善的中文回答

請保持回答簡潔但有深度，大約 2-3 段文字。特別關注用戶選取的文字部分。"""
        else:
            system_prompt = """你是一個專業的閱讀助手，專門幫助讀者理解和討論書籍內容。
你的任務是：
1. 回答關於書籍內容的問題
2. 提供深入的分析和見解
3. 用清晰、友善的中文回答
4. 如果問題與當前章節無關，可以根據書籍元數據提供一般性的回答

請保持回答簡潔但有深度，大約 2-3 段文字。"""

        full_prompt = f"{system_prompt}\n\n{book_context}\n\n用戶問題：{chat_request.message}"

        # Generate response
        response = model.generate_content(full_prompt)

        return JSONResponse(content={
            "response": response.text,
            "chapter_info": {
                "index": chat_request.chapter_index,
                "title": current_chapter.title,
                "total": len(book.spine)
            }
        })

    except Exception as e:
        print(f"Chat error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"處理請求時發生錯誤：{str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8123")
    uvicorn.run(app, host="127.0.0.1", port=8123)
