# --- Check and install charset_normalizer if needed ---
import sys
import subprocess

try:
    import charset_normalizer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "charset_normalizer"])
    import charset_normalizer

# --- Imports ---
import pygame
import tkinter as tk
from tkinter import filedialog
import json
import os
import numpy as np

# Initialize Tkinter root (hidden)
root = tk.Tk()
root.withdraw()

pygame.init()
pygame.mixer.init()

WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PyGame Music Player with Visualizer & Playlist")

font = pygame.font.SysFont('arial', 22)
small_font = pygame.font.SysFont('arial', 16)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
LIGHT_GRAY = (230, 230, 230)
RED = (200, 30, 30)
BLUE = (50, 120, 220)

BUTTON_COLOR = (100, 149, 237)
BUTTON_HOVER = (70, 130, 180)

# Set base background color to charcoal grey
CHARCOAL_GREY = (30, 30, 30)

songs = []
current_index = 0
use_mp3 = True
playing = False
error_msg = ""

channel = pygame.mixer.Channel(0)
current_sound = None

# For tracking elapsed time
start_ticks = 0  # pygame.time.get_ticks() when song starts
paused_ticks = 0
paused = False

# Scroll for playlist
playlist_scroll = 0
playlist_item_height = 30

class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER
        self.is_hovered = False

    def draw(self, surface):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        txt_surf = font.render(self.text, True, WHITE)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

def load_json():
    global songs, current_index, playing, error_msg, paused, paused_ticks, start_ticks, playlist_scroll
    error_msg = ""
    paused = False
    paused_ticks = 0
    start_ticks = 0
    playlist_scroll = 0
    file_path = filedialog.askopenfilename(
        title="Select JSON file",
        filetypes=[("JSON Files", "*.json")]
    )
    if not file_path:
        error_msg = "No file selected."
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            songs = json.load(f)
        if not songs:
            error_msg = "JSON file is empty."
            return
        current_index = 0
        playing = False
        play_song()
    except Exception as e:
        error_msg = f"Failed to load JSON: {e}"

def play_song():
    global current_sound, playing, error_msg, start_ticks, paused, paused_ticks
    error_msg = ""
    paused = False
    paused_ticks = 0
    if not songs:
        error_msg = "No songs loaded."
        return
    song = songs[current_index]
    audio_path = song['mp3'] if use_mp3 else song['wav']
    if not os.path.exists(audio_path):
        error_msg = f"File not found:\n{audio_path}"
        playing = False
        return
    try:
        if channel.get_busy():
            channel.stop()
        current_sound = pygame.mixer.Sound(audio_path)
        channel.play(current_sound)
        playing = True
        start_ticks = pygame.time.get_ticks()
    except Exception as e:
        error_msg = f"Playback error:\n{e}"
        playing = False

def stop_song():
    global playing, paused
    channel.stop()
    playing = False
    paused = False

def pause_song():
    global playing, paused, paused_ticks
    if channel.get_busy():
        channel.pause()
        playing = False
        paused = True
        paused_ticks = pygame.time.get_ticks()

def unpause_song():
    global playing, paused, start_ticks, paused_ticks
    channel.unpause()
    playing = True
    paused = False
    pause_duration = pygame.time.get_ticks() - paused_ticks
    start_ticks += pause_duration

def toggle_play_pause():
    if channel.get_busy() and playing:
        pause_song()
    else:
        unpause_song()

def next_song():
    global current_index
    if not songs:
        return
    current_index = (current_index + 1) % len(songs)
    play_song()

def prev_song():
    global current_index
    if not songs:
        return
    current_index = (current_index - 1) % len(songs)
    play_song()

def toggle_audio_type():
    global use_mp3
    use_mp3 = not use_mp3
    if songs and playing:
        play_song()

button_width = 110
button_height = 40
button_padding = 15
start_x = 20
start_y = HEIGHT - button_height - 20

buttons = {
    "load": Button((start_x, start_y, button_width, button_height), "Load JSON"),
    "prev": Button((start_x + (button_width + button_padding), start_y, button_width, button_height), "Previous"),
    "play_pause": Button((start_x + 2*(button_width + button_padding), start_y, button_width, button_height), "Play/Pause"),
    "next": Button((start_x + 3*(button_width + button_padding), start_y, button_width, button_height), "Next"),
    "toggle": Button((start_x + 4*(button_width + button_padding), start_y, button_width + 40, button_height), "Toggle MP3/WAV"),
}

def draw_text_wrapped(surface, text, font, color, x, y, max_width, line_spacing=4):
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + ("" if current_line == "" else " ") + word
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    for line in lines:
        text_surf = font.render(line, True, color)
        surface.blit(text_surf, (x, y))
        y += text_surf.get_height() + line_spacing
    return y

# Viridis-like gradient colors precomputed for performance
_VIRIDIS_COLORS = [
    (68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142),
    (33, 144, 141), (39, 173, 129), (92, 200, 99), (170, 220, 50),
    (253, 231, 37)
]

def viridis_color(value):
    """Map normalized value [0,1] to viridis color gradient"""
    if value <= 0:
        return _VIRIDIS_COLORS[0]
    if value >= 1:
        return _VIRIDIS_COLORS[-1]
    scaled = value * (len(_VIRIDIS_COLORS) - 1)
    i = int(scaled)
    f = scaled - i
    c1 = _VIRIDIS_COLORS[i]
    c2 = _VIRIDIS_COLORS[min(i+1, len(_VIRIDIS_COLORS)-1)]
    r = int(c1[0] + (c2[0] - c1[0]) * f)
    g = int(c1[1] + (c2[1] - c1[1]) * f)
    b = int(c1[2] + (c2[2] - c1[2]) * f)
    return (r, g, b)

def draw_visualizer(surface, sound, elapsed_ms, x, y, width, height):
    if not sound:
        return
    try:
        arr = pygame.sndarray.array(sound)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        total_samples = arr.shape[0]

        length_sec = sound.get_length()
        elapsed_sec = elapsed_ms / 1000
        if elapsed_sec > length_sec:
            elapsed_sec = length_sec
        current_sample = int((elapsed_sec / length_sec) * total_samples)

        window_size = min(5000, total_samples)
        half_win = window_size // 2
        start_idx = max(0, current_sample - half_win)
        end_idx = min(total_samples, start_idx + window_size)
        window = arr[start_idx:end_idx]

        max_val = np.max(np.abs(window)) if np.max(np.abs(window)) != 0 else 1
        normalized = window / max_val

        # Parameters for smaller grid boxes
        grid_cols = 30  # smaller grid horizontally
        grid_rows = 15  # smaller grid vertically
        box_w = width / grid_cols
        box_h = height / grid_rows

        samples_per_box = len(normalized) // (grid_cols * grid_rows)
        if samples_per_box == 0:
            samples_per_box = 1

        # We'll create a 2D grid of boxes; for each box average the amplitude from the corresponding samples
        for row in range(grid_rows):
            for col in range(grid_cols):
                idx_start = (row * grid_cols + col) * samples_per_box
                idx_end = idx_start + samples_per_box
                if idx_end > len(normalized):
                    idx_end = len(normalized)
                if idx_start >= len(normalized):
                    avg = 0
                else:
                    segment = normalized[idx_start:idx_end]
                    avg = np.abs(segment).mean() if len(segment) > 0 else 0

                # Apply dB-like log scale for color intensity
                log_avg = np.log10(1 + 9 * avg)  # [0,1] log scaled

                color = viridis_color(log_avg)

                rect_x = x + col * box_w
                rect_y = y + row * box_h
                pygame.draw.rect(surface, color, (rect_x, rect_y, box_w - 2, box_h - 2), border_radius=3)

    except Exception:
        pass

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def draw_playlist(surface, x, y, width, height):
    global playlist_scroll
    pygame.draw.rect(surface, LIGHT_GRAY, (x, y, width, height))
    max_visible = height // playlist_item_height

    if len(songs) > max_visible:
        # Clamp scroll
        playlist_scroll = max(0, min(playlist_scroll, len(songs) - max_visible))
    else:
        playlist_scroll = 0

    visible_songs = songs[playlist_scroll:playlist_scroll + max_visible]

    for i, song in enumerate(visible_songs):
        idx = i + playlist_scroll
        item_rect = pygame.Rect(x, y + i * playlist_item_height, width, playlist_item_height)
        if idx == current_index:
            pygame.draw.rect(surface, BLUE, item_rect)
        else:
            pygame.draw.rect(surface, WHITE, item_rect)

        title = song.get('title', 'Unknown Title')
        artist = song.get('artist', 'Unknown Artist')
        display_text = f"{idx + 1}. {title} - {artist}"
        text_surf = small_font.render(display_text, True, BLACK)
        surface.blit(text_surf, (x + 5, y + i * playlist_item_height + 5))

def handle_playlist_click(pos, x, y, width, height):
    global current_index, playing
    if not songs:
        return
    if x <= pos[0] <= x + width and y <= pos[1] <= y + height:
        relative_y = pos[1] - y
        idx = relative_y // playlist_item_height + playlist_scroll
        if idx < len(songs):
            current_index = idx
            play_song()

def main():
    global playing, error_msg, playlist_scroll

    clock = pygame.time.Clock()
    running = True

    # Playlist area
    playlist_x = WIDTH - 300 - 20
    playlist_y = 20
    playlist_w = 300
    playlist_h = HEIGHT - 100

    while running:
        screen.fill(CHARCOAL_GREY)
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    for btn in buttons.values():
                        if btn.is_clicked((mx, my)):
                            if btn == buttons['load']:
                                load_json()
                            elif btn == buttons['play_pause']:
                                if songs:
                                    toggle_play_pause()
                            elif btn == buttons['next']:
                                next_song()
                            elif btn == buttons['prev']:
                                prev_song()
                            elif btn == buttons['toggle']:
                                toggle_audio_type()
                    handle_playlist_click((mx, my), playlist_x, playlist_y, playlist_w, playlist_h)
                elif event.button == 4:  # Scroll up
                    playlist_scroll = max(0, playlist_scroll - 1)
                elif event.button == 5:  # Scroll down
                    playlist_scroll = min(max(0, len(songs) - (playlist_h // playlist_item_height)), playlist_scroll + 1)

            elif event.type == pygame.MOUSEWHEEL:
                # Optional: Support mouse wheel event if needed, but pygame.MOUSEBUTTONDOWN scroll works
                pass

        for btn in buttons.values():
            btn.check_hover((mx, my))
            btn.draw(screen)

        # Display current song info
        info_x = 20
        info_y = 20
        info_w = WIDTH - 360

        if songs:
            current_song = songs[current_index]
            title = current_song.get('title', 'Unknown Title')
            artist = current_song.get('artist', 'Unknown Artist')
            display_str = f"Now Playing: {title} - {artist} ({'MP3' if use_mp3 else 'WAV'})"
        else:
            display_str = "No songs loaded."

        draw_text_wrapped(screen, display_str, font, WHITE, info_x, info_y, info_w)

        # Draw visualizer grid boxes smaller with viridis dB color
        if playing:
            elapsed_ms = pygame.time.get_ticks() - start_ticks
        else:
            elapsed_ms = 0
        draw_visualizer(screen, current_sound, elapsed_ms, info_x, info_y + 60, info_w, 150)

        # Draw playlist
        draw_playlist(screen, playlist_x, playlist_y, playlist_w, playlist_h)

        # Display error message if any
        if error_msg:
            err_rect = pygame.Rect(20, HEIGHT - 80, WIDTH - 40, 60)
            pygame.draw.rect(screen, RED, err_rect, border_radius=8)
            draw_text_wrapped(screen, error_msg, small_font, WHITE, err_rect.x + 10, err_rect.y + 10, err_rect.width - 20)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
