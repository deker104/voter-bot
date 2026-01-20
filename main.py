import asyncio
import io
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiosqlite
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.types.input_file import BufferedInputFile
from aiogram.exceptions import TelegramBadRequest

import matplotlib
matplotlib.use("Agg")  # важно для серверов без дисплея
from matplotlib import pyplot as plt

from graph import plot_irv_pairwise_matrix

# =========================
# Конфиг вариантов (15 шт.)
# =========================

@dataclass(frozen=True)
class OptionSeed:
    id: int
    title: str
    caption: str
    image_path: str  # локальный путь (для первого аплоада -> получения file_id)


# Замените title/caption под свои нужды.
# image_path должен существовать (assets/1.png ... assets/15.png)
DEFAULT_OPTIONS: List[OptionSeed] = [
    OptionSeed(1,  "Скетч 1",  "by @yoyomif & @neuroblin",  "assets/1.png"),
    OptionSeed(2,  "Скетч 2",  "by @yoyomif & @neuroblin",  "assets/2.png"),
    OptionSeed(3,  "Скетч 3",  "by @yoyomif & @neuroblin",  "assets/3.png"),
    OptionSeed(4,  "Скетч 4",  "by @yoyomif & @neuroblin",  "assets/4.png"),
    OptionSeed(5,  "Скетч 5",  "by @yoyomif & @neuroblin",  "assets/5.png"),
    OptionSeed(6,  "Скетч 6",  "by @yoyomif & @neuroblin",  "assets/6.png"),
    OptionSeed(7,  "Скетч 7",  "by @yoyomif & @neuroblin",  "assets/7.png"),
    OptionSeed(8,  "Скетч 8",  "by @yoyomif & @neuroblin",  "assets/8.png"),
    OptionSeed(9,  "Скетч 9",  "by @yoyomif & @neuroblin",  "assets/9.png"),
    OptionSeed(10, "Скетч 10", "by @yoyomif & @neuroblin", "assets/10.png"),
    OptionSeed(11, "Скетч 11", "by @yoyomif & @neuroblin", "assets/11.png"),
    OptionSeed(12, "Скетч 12", "by @yoyomif & @neuroblin", "assets/12.png"),
    OptionSeed(13, "Скетч 13", "by @yoyomif & @neuroblin", "assets/13.png"),
    OptionSeed(14, "Скетч 14", "by @yoyomif & @neuroblin", "assets/14.png"),
    OptionSeed(15, "Скетч 15", "by @yoyomif & @neuroblin", "assets/15.png"),
]

ADMIN_IDS = list(map(int, os.getenv("ADMIN_IDS", "").split(",")))

# =========================
# SQLite слой
# =========================

class Database:
    def __init__(self, path: str = "bot.sqlite3") -> None:
        self.path = path
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute("PRAGMA foreign_keys=ON;")

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()

    async def init(self) -> None:
        assert self.conn is not None

        await self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS options (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                caption TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_file_id TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                user_id INTEGER PRIMARY KEY,
                selected_json TEXT NOT NULL,
                unselected_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ballots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT,
                ranking_json TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                is_single_vote BOOLEAN DEFAULT 0
            );
            """
        )

        # Засеять options, если ещё нет
        for opt in DEFAULT_OPTIONS:
            await self.conn.execute(
                """
                INSERT OR IGNORE INTO options (id, title, caption, image_path, image_file_id)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (opt.id, opt.title, opt.caption, opt.image_path),
            )
        await self.conn.commit()

    async def get_options(self) -> List[aiosqlite.Row]:
        assert self.conn is not None
        cur = await self.conn.execute("SELECT * FROM options ORDER BY id ASC")
        rows = await cur.fetchall()
        return rows

    async def get_options_map(self) -> Dict[int, aiosqlite.Row]:
        rows = await self.get_options()
        return {int(r["id"]): r for r in rows}

    async def update_option_file_id(self, option_id: int, file_id: str) -> None:
        assert self.conn is not None
        await self.conn.execute(
            "UPDATE options SET image_file_id = ? WHERE id = ?",
            (file_id, option_id),
        )
        await self.conn.commit()

    async def upsert_session(self, user_id: int, selected: List[int], unselected: List[int]) -> None:
        assert self.conn is not None
        now = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            """
            INSERT INTO sessions (user_id, selected_json, unselected_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                selected_json = excluded.selected_json,
                unselected_json = excluded.unselected_json
            """,
            (user_id, json.dumps(selected), json.dumps(unselected), now),
        )
        await self.conn.commit()

    async def get_session(self, user_id: int) -> Optional[Tuple[List[int], List[int]]]:
        assert self.conn is not None
        cur = await self.conn.execute(
            "SELECT selected_json, unselected_json FROM sessions WHERE user_id = ?",
            (user_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        selected = json.loads(row["selected_json"])
        unselected = json.loads(row["unselected_json"])
        return (selected, unselected)

    async def delete_session(self, user_id: int) -> None:
        assert self.conn is not None
        await self.conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        await self.conn.commit()

    async def add_ballot(self, user_id: int, username: Optional[str], ranking: List[int], is_single_vote: bool = False) -> None:
        assert self.conn is not None
        now = datetime.now(timezone.utc).isoformat()
        await self.conn.execute(
            """
            INSERT INTO ballots (user_id, username, ranking_json, submitted_at, is_single_vote)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, username, json.dumps(ranking), now, int(is_single_vote)),
        )
        await self.conn.commit()

    async def get_all_ballots(self) -> List[List[int]]:
        assert self.conn is not None
        # Берем только последний бюллетень от каждого пользователя
        query = """
            SELECT ranking_json
            FROM ballots b1
            WHERE id = (
                SELECT MAX(id)
                FROM ballots b2
                WHERE b2.user_id = b1.user_id
            )
            ORDER BY id ASC
        """
        cur = await self.conn.execute(query)
        rows = await cur.fetchall()
        return [json.loads(r["ranking_json"]) for r in rows]

    async def get_last_ballot(self, user_id: int) -> Optional[List[int]]:
        assert self.conn is not None
        cur = await self.conn.execute(
            "SELECT ranking_json FROM ballots WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,)
        )
        row = await cur.fetchone()
        if row:
            return json.loads(row["ranking_json"])
        return None


# =========================
# Клавиатура/текст
# =========================

def build_poll_text(options_by_id: Dict[int, aiosqlite.Row], selected: List[int]) -> str:
    lines = [
        "<b>Соберите рейтинг концептов</b>",
        "",
        "Нажимайте варианты в порядке предпочтения:",
        "• выбранные добавляются в конец списка выбранных",
        "• повторное нажатие снимает выбор",
        "• нужно выбрать хотя бы один вариант",
        "",
    ]
    if selected:
        lines.append("<b>Ваш текущий порядок (1 — лучший):</b>")
        for i, oid in enumerate(selected, start=1):
            title = options_by_id[oid]["title"]
            lines.append(f"{i}. {title}")
    else:
        lines.append("<i>Пока ничего не выбрано.</i>")

    lines.append("")
    lines.append("Когда закончите — нажмите <b>«Отправить»</b>.")
    return "\n".join(lines)

def build_keyboard(options_by_id: Dict[int, aiosqlite.Row], selected: List[int], unselected: List[int]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []

    # Блок выбранных (вверху)
    for rank, oid in enumerate(selected, start=1):
        title = options_by_id[oid]["title"]
        text = f"✅ {rank}. {title}"
        rows.append([InlineKeyboardButton(text=text, callback_data=f"pick:{oid}")])

    # Блок невыбранных (внизу)
    for oid in unselected:
        title = options_by_id[oid]["title"]
        text = f"▫️ {title}"
        rows.append([InlineKeyboardButton(text=text, callback_data=f"pick:{oid}")])

    # Кнопка отправки
    rows.append([InlineKeyboardButton(text="📩 Отправить", callback_data="submit")])

    return InlineKeyboardMarkup(inline_keyboard=rows)

def build_single_choice_keyboard(options_by_id: Dict[int, aiosqlite.Row]) -> InlineKeyboardMarkup:
    rows: List[List[InlineKeyboardButton]] = []
    for oid, row in options_by_id.items():
        title = row["title"]
        rows.append([InlineKeyboardButton(text=title, callback_data=f"vote_one:{oid}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

# =========================
# Handlers
# =========================

dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(message: Message, bot: Bot, db: Database) -> None:
    # Проверим, что картинки на месте (если используете локальные пути)
    # Если хотите — удалите проверку.
    for opt in DEFAULT_OPTIONS:
        if not os.path.exists(opt.image_path):
            await message.answer(
                "Не найдена картинка: "
                f"<code>{opt.image_path}</code>\n"
                "Положите 15 файлов в папку assets/ (1.png ... 15.png) "
                "или поменяйте пути в DEFAULT_OPTIONS."
            )
            return

    # Сбросить предыдущую сессию (если была)
    await db.delete_session(message.from_user.id)

    options = await db.get_options()

    # 1) Отправляем 15 сообщений (фото + текст).
    #    Важно: после первого отправления мы сохраняем image_file_id в sqlite,
    #    и дальше Telegram не будет требовать повторной загрузки файла.
    for row in options:
        option_id = int(row["id"])
        title = row["title"]
        caption = row["caption"]
        image_path = row["image_path"]
        file_id = row["image_file_id"]

        caption_full = f"<b>{title}</b>\n{caption}"

        if file_id:
            # уже закешировано на стороне Telegram
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=file_id,
                caption=caption_full,
                parse_mode=ParseMode.HTML,
            )
        else:
            # первый раз: грузим с диска, получаем file_id, сохраняем в sqlite
            msg = await bot.send_photo(
                chat_id=message.chat.id,
                photo=FSInputFile(image_path),
                caption=caption_full,
                parse_mode=ParseMode.HTML,
            )
            # Telegram возвращает несколько размеров, берём самый большой
            new_file_id = msg.photo[-1].file_id
            await db.update_option_file_id(option_id, new_file_id)

        # маленькая пауза, чтобы не упереться в лимиты на очень загруженных ботах
        await asyncio.sleep(0.05)

    # 2) Предлагаем выбрать один лучший вариант
    options_by_id = {int(r["id"]): r for r in options}
    text = (
        "<b>Выберите один лучший вариант</b>\n\n"
        "Пожалуйста, нажмите на кнопку с названием варианта, который вам нравится больше всего."
    )
    kb = build_single_choice_keyboard(options_by_id)
    await message.answer(text, reply_markup=kb, parse_mode=ParseMode.HTML)


@dp.callback_query(F.data.startswith("vote_one:"))
async def on_vote_one(callback: CallbackQuery, db: Database) -> None:
    assert callback.message is not None

    try:
        option_id = int(callback.data.split(":")[1])
    except Exception:
        await callback.answer("Ошибка данных.", show_alert=True)
        return

    options_by_id = await db.get_options_map()
    title = options_by_id[option_id]["title"]

    text = f"Вы выбрали <b>«{title}»</b>.\nПодтверждаете выбор?"
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Да", callback_data=f"confirm_vote_one:{option_id}"),
            InlineKeyboardButton(text="🔙 Нет", callback_data="cancel_vote_one")
        ]
    ])
    await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
    await callback.answer()


@dp.callback_query(F.data.startswith("confirm_vote_one:"))
async def on_confirm_vote_one(callback: CallbackQuery, db: Database) -> None:
    assert callback.message is not None
    user_id = callback.from_user.id

    try:
        option_id = int(callback.data.split(":")[1])
    except Exception:
        await callback.answer("Ошибка данных.", show_alert=True)
        return

    # Сохраняем голос (как список из одного элемента)
    await db.add_ballot(
        user_id=user_id,
        username=callback.from_user.username,
        ranking=[option_id],
        is_single_vote=True,
    )

    options_by_id = await db.get_options_map()
    title = options_by_id[option_id]["title"]

    text = (
        f"✅ Ваш голос за <b>«{title}»</b> принят!\n\n"
        "Хотите составить подробный рейтинг остальных вариантов?\n"
        "Это поможет нам лучше учесть ваши предпочтения в случае, если выбранный вами вариант не победит."
    )

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Составить рейтинг", callback_data="start_ranking")]
    ])

    await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
    await callback.answer("Голос принят!")


@dp.callback_query(F.data == "cancel_vote_one")
async def on_cancel_vote_one(callback: CallbackQuery, db: Database) -> None:
    assert callback.message is not None

    options = await db.get_options()
    options_by_id = {int(r["id"]): r for r in options}

    text = (
        "<b>Выберите один лучший вариант</b>\n\n"
        "Пожалуйста, нажмите на кнопку с названием варианта, который вам нравится больше всего."
    )
    kb = build_single_choice_keyboard(options_by_id)
    await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
    await callback.answer()


@dp.callback_query(F.data == "start_ranking")
async def on_start_ranking(callback: CallbackQuery, db: Database) -> None:
    assert callback.message is not None
    user_id = callback.from_user.id

    # Получаем последний голос, чтобы узнать, что пользователь выбрал первым
    last_ranking = await db.get_last_ballot(user_id)
    selected = []
    if last_ranking:
        selected = last_ranking  # там должен быть [option_id]

    options = await db.get_options()
    all_ids = [int(r["id"]) for r in options]
    
    # Формируем unselected (все кроме уже выбранного)
    unselected = [x for x in all_ids if x not in selected]
    secrets.SystemRandom().shuffle(unselected)

    # Создаем сессию
    await db.upsert_session(user_id, selected, unselected)

    options_by_id = {int(r["id"]): r for r in options}
    text = build_poll_text(options_by_id, selected)
    kb = build_keyboard(options_by_id, selected, unselected)

    await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
    await callback.answer()


@dp.callback_query(F.data.startswith("pick:"))
async def on_pick(callback: CallbackQuery, db: Database) -> None:
    assert callback.message is not None
    user_id = callback.from_user.id

    try:
        option_id = int(callback.data.split(":")[1])
    except Exception:
        await callback.answer("Ошибка данных.", show_alert=True)
        return

    logging.debug(f"pick: {option_id}")

    session = await db.get_session(user_id)
    options_by_id = await db.get_options_map()
    all_ids = list(options_by_id.keys())

    if session is None:
        # если сессия потерялась (рестарт бота), создадим новую
        unselected = all_ids[:]
        secrets.SystemRandom().shuffle(unselected)
        selected: List[int] = []
    else:
        selected, unselected = session

    if option_id in selected:
        # 5) повторное нажатие: снять выбор, переместить вниз (в начало невыбранных)
        selected = [x for x in selected if x != option_id]
        # на всякий случай уберём из unselected если вдруг там уже есть
        unselected = [x for x in unselected if x != option_id]
        unselected.insert(0, option_id)
    else:
        # 4) выбрать: переместить наверх (в конец выбранных)
        unselected = [x for x in unselected if x != option_id]
        # на всякий случай уберём из selected если вдруг там уже есть
        selected = [x for x in selected if x != option_id]
        selected.append(option_id)

    await db.upsert_session(user_id, selected, unselected)

    text = build_poll_text(options_by_id, selected)
    kb = build_keyboard(options_by_id, selected, unselected)

    try:
        await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
    except TelegramBadRequest as e:
        # Частая причина: "message is not modified"
        if "message is not modified" not in str(e).lower():
            raise

    await callback.answer()


@dp.callback_query(F.data == "submit")
async def on_submit(callback: CallbackQuery, db: Database) -> None:
    assert callback.message is not None
    user_id = callback.from_user.id

    session = await db.get_session(user_id)
    if session is None:
        await callback.answer("Сессия не найдена. Нажмите /start заново.", show_alert=True)
        return

    selected, unselected = session

    if not selected:
        await callback.answer("Сначала выберите хотя бы один вариант.", show_alert=True)
        return

    options_by_id = await db.get_options_map()
    human = [options_by_id[i]["title"] for i in selected]

    text = (
        "<b>Ваш рейтинг:</b>\n"
        + "\n".join([f"{i+1}. {t}" for i, t in enumerate(human)])
        + "\n\nПодтверждаете отправку?"
    )

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Да", callback_data="confirm_submit"),
            InlineKeyboardButton(text="🔙 Нет", callback_data="cancel_submit")
        ]
    ])
    await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
    await callback.answer()


@dp.callback_query(F.data == "confirm_submit")
async def on_confirm_submit(callback: CallbackQuery, db: Database) -> None:
    assert callback.message is not None
    user_id = callback.from_user.id

    session = await db.get_session(user_id)
    if session is None:
        await callback.answer("Сессия не найдена. Нажмите /start заново.", show_alert=True)
        return

    selected, unselected = session
    # В IRV обычно важен порядок предпочтений.
    # Здесь ranking = выбранные (в порядке ранга 1..N). Невыбранные не попадают в бюллетень (будут “exhausted”).
    ranking = selected[:]

    await db.add_ballot(
        user_id=user_id,
        username=callback.from_user.username,
        ranking=ranking,
    )
    await db.delete_session(user_id)

    # Убираем клавиатуру и подтверждаем
    options_by_id = await db.get_options_map()
    human = [options_by_id[i]["title"] for i in ranking]
    text = (
        "✅ <b>Голос принят!</b>\n\n"
        "<b>Ваш рейтинг:</b>\n"
        + "\n".join([f"{i+1}. {t}" for i, t in enumerate(human)])
    )
    await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=None)
    await callback.answer("Сохранено ✅")


@dp.callback_query(F.data == "cancel_submit")
async def on_cancel_submit(callback: CallbackQuery, db: Database) -> None:
    assert callback.message is not None
    user_id = callback.from_user.id

    session = await db.get_session(user_id)
    if session is None:
        await callback.answer("Сессия не найдена. Нажмите /start заново.", show_alert=True)
        return

    selected, unselected = session
    options_by_id = await db.get_options_map()

    text = build_poll_text(options_by_id, selected)
    kb = build_keyboard(options_by_id, selected, unselected)

    await callback.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
    await callback.answer()

@dp.message(Command("graph"))
async def cmd_graph(message: Message, db: Database) -> None:
    user_id = message.from_user.id
    if user_id not in ADMIN_IDS:
        await message.answer("Ты не админ.")
        return

    ballots = await db.get_all_ballots()
    if not ballots:
        await message.answer("Пока нет ни одного голоса.")
        return

    options_by_id = await db.get_options_map()
    candidate_ids = sorted(options_by_id.keys())

    # строим график (твоя функция из прошлых сообщений)
    fig, ax = plot_irv_pairwise_matrix(
        ballots,
        candidate_ids=candidate_ids,
        labels=None,
        figsize=(13, 11),
    )

    # сохраняем в память
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)  # важно, чтобы не текла память
    buf.seek(0)

    photo = BufferedInputFile(buf.getvalue(), filename="irv_matrix.png")
    await message.answer_photo(photo=photo, caption=f"График по {len(ballots)} рейтингам")


# =========================
# Entrypoint
# =========================

async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Нужно установить переменную окружения BOT_TOKEN")

    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    db = Database("bot.sqlite3")
    await db.connect()
    await db.init()

    try:
        await dp.start_polling(bot, db=db)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
