"""On-demand animated GIF of wrist-IMU signals; Pillow ships with Streamlit.

Research visualization, not measured full-body pose tracking.
Author: Kevin Sun | Mentor: Dr. Qingyang Xiao.
"""
from __future__ import annotations

from io import BytesIO
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def swimmer_gif(session_data: pd.DataFrame, windows: pd.DataFrame, seconds: int = 7, fps: int = 9) -> bytes:
    if session_data.empty or windows.empty:
        raise ValueError('A session with AI predictions is required.')
    sorted_data = session_data.sort_values('timestamp').reset_index(drop=True)
    window_predictions = windows[['window_start', 'predicted_stroke', 'prediction_confidence']].sort_values('window_start')
    duration = min(len(sorted_data), seconds * 20)
    frames_count = seconds * fps
    chosen = sorted_data.iloc[np.linspace(0, duration - 1, frames_count).astype(int)].copy()
    chosen = pd.merge_asof(chosen.sort_values('timestamp'), window_predictions,
                           left_on='timestamp', right_on='window_start', direction='nearest')
    font = ImageFont.load_default()
    frames = []
    W, H = 790, 320
    for i, row in enumerate(chosen.itertuples(index=False)):
        image = Image.new('RGB', (W, H), '#07243c')
        draw = ImageDraw.Draw(image)
        for y in range(92, H, 30):
            wave = [(x, int(y + 5 * math.sin(x/36 + i/5))) for x in range(0, W, 12)]
            draw.line(wave, fill='#165578', width=2)
        draw.line([(0, 70), (W, 70)], fill='#52bed9', width=3)
        gx = float(np.nan_to_num(getattr(row, 'gyro_x', 0.0)))
        gz = float(np.nan_to_num(getattr(row, 'gyro_z', 0.0)))
        ay = float(np.nan_to_num(getattr(row, 'acc_y', 0.0)))
        phase = 2 * math.pi * i / 22
        cy = 188 + 7 * math.sin(phase)
        shoulder = (360, cy - 10)
        hip = (495, cy + 6)
        draw.line([shoulder, hip], fill='#e4f7ff', width=14)
        draw.ellipse((302, cy-34, 348, cy+12), fill='#f6c785', outline='#effbff', width=2)
        a1 = math.pi + .55 * math.tanh(gz/150) + .24 * math.sin(phase)
        a2 = math.pi - .55 * math.tanh(gx/130) - .24 * math.sin(phase)
        for a in (a1, a2):
            end = (shoulder[0] + 95*math.cos(a), shoulder[1] + 95*math.sin(a))
            draw.line([shoulder, end], fill='#e6f6ff', width=8)
        kick = .34 * math.tanh(ay/.5) + .15*math.sin(2*phase)
        for sign in [-1, 1]:
            draw.line([hip, (635, cy+17+sign*kick*65)], fill='#d9f6ff', width=9)
        stroke = str(getattr(row, 'predicted_stroke', 'unknown')).title()
        confidence = float(np.nan_to_num(getattr(row, 'prediction_confidence', 0)))
        hr = float(np.nan_to_num(getattr(row, 'heart_rate_bpm', 0)))
        draw.text((20, 22), 'AquaMind | AI swimming replay', font=font, fill='#dcfaff')
        draw.text((20, 115), f'Stroke: {stroke}', font=font, fill='#ffffff')
        draw.text((20, 137), f'Model confidence: {confidence:.0%}', font=font, fill='#ffffff')
        draw.text((20, 159), f'Watch HR: {hr:.0f} bpm' if hr > 0 else 'Watch HR: unavailable', font=font, fill='#ffffff')
        draw.text((20, 280), 'Kevin Sun  |  Mentor: Dr. Qingyang Xiao', font=font, fill='#b2e5f9')
        frames.append(image)
    output = BytesIO()
    frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:],
                   duration=int(1000/fps), loop=0, optimize=True)
    return output.getvalue()
