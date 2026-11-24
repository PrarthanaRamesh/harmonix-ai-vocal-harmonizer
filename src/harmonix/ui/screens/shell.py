"""State-driven Pygame screens for the Item 10 functional UI."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from .. import theme
from ..state_binding import AppViewModel, upload_progress_label


@dataclass
class ShellLayout:
    controls: dict[str, pygame.Rect]
    enabled: set[str]


def render_shell(
    surface: pygame.Surface,
    view_model: AppViewModel,
    *,
    focused: str,
    hovered: str | None,
    upload_voices: tuple[str, ...],
    upload_path_entry: str,
    upload_start_error: str | None,
    settings_warning: str | None = None,
    settings_error: str | None = None,
    settings_retry_pending: bool = False,
    session_end_error: str | None = None,
    session_end_pending: bool = False,
    discard_confirmation_pending: bool = False,
    close_confirmation_pending: bool = False,
    export_toast: str | None = None,
    export_ui_error: str | None = None,
    input_devices: tuple[object, ...] = (),
    output_devices: tuple[object, ...] = (),
    input_device_selector: str | None = None,
    output_device_selector: str | None = None,
    audio_device_error: str | None = None,
    audio_devices_loaded: bool = False,
    live_start_error: str | None = None,
    live_key: object | None = None,
) -> ShellLayout:
    surface.fill(theme.BACKGROUND)
    width, height = surface.get_size()
    margin = max(24, min(48, width // 24))
    panel = pygame.Rect(margin, margin, max(1, width - 2 * margin), max(1, height - 2 * margin))
    pygame.draw.rect(surface, theme.SURFACE, panel, border_radius=12)
    pygame.draw.rect(surface, theme.BORDER, panel, width=1, border_radius=12)
    controls: dict[str, pygame.Rect] = {}
    enabled: set[str] = set()
    fonts = (pygame.font.Font(None, 42), pygame.font.Font(None, 27), pygame.font.Font(None, 22))

    def label(text: str, position: tuple[int, int], color: tuple[int, int, int] = theme.TEXT, font: pygame.font.Font | None = None) -> None:
        rendered = (font or fonts[1]).render(text, True, color)
        surface.blit(rendered, position)

    def button(name: str, text: str, rect: pygame.Rect, *, disabled: bool = False, primary: bool = False) -> None:
        controls[name] = rect
        if not disabled:
            enabled.add(name)
        if disabled:
            color = tuple(int(channel * 0.4) for channel in theme.BORDER)
            text_color = tuple(int(channel * 0.55) for channel in theme.MUTED_TEXT)
        elif name == hovered:
            color = theme.ACCENT
            text_color = theme.BACKGROUND
        elif primary:
            color = theme.PRIMARY
            text_color = theme.TEXT
        else:
            color = theme.BORDER
            text_color = theme.TEXT
        pygame.draw.rect(surface, color, rect, border_radius=8)
        if name == focused and not disabled:
            pygame.draw.rect(surface, theme.ACCENT, rect.inflate(6, 6), width=2, border_radius=9)
        text_surface = fonts[2].render(text, True, text_color)
        surface.blit(text_surface, text_surface.get_rect(center=rect.center))

    label("HARMONIX", (panel.left + 28, panel.top + 20), theme.TEXT, fonts[0])
    label(view_model.screen_name.replace("_", " ").upper(), (panel.left + 30, panel.top + 67), theme.ACCENT, fonts[2])
    label("LIVE", (panel.right - 196, panel.top + 31), theme.MUTED_TEXT, fonts[2])
    label("UPLOAD", (panel.right - 112, panel.top + 31), theme.MUTED_TEXT, fonts[2])
    button("nav_live", "LIVE", pygame.Rect(panel.right - 208, panel.top + 20, 76, 34))
    button("nav_upload", "UPLOAD", pygame.Rect(panel.right - 120, panel.top + 20, 92, 34))
    if session_end_pending:
        button("retry_session_end", "RETRY SAVE & END", pygame.Rect(panel.right - 244, panel.top + 62, 216, 36), primary=True)
    if session_end_error:
        label(f"Session kept. Recent-history save failed: {session_end_error}", (panel.left + 30, panel.top + 94), theme.ERROR, fonts[2])
    content_left = panel.left + 30
    content_top = panel.top + 115
    content_width = panel.width - 60
    session = view_model.session
    screen = view_model.screen_name

    if screen == "home":
        label("Make space for every voice.", (content_left, content_top), theme.TEXT, fonts[0])
        if upload_start_error:
            label(f"Could not reopen recent session: {upload_start_error}", (content_left, content_top + 46), theme.ERROR, fonts[2])
        card_width = max(180, (content_width - 20) // 2)
        button("choose_live", "LIVE HARMONIZE", pygame.Rect(content_left, content_top + 76, card_width, 92), primary=True)
        button("choose_upload", "UPLOAD & PLAY", pygame.Rect(content_left + card_width + 20, content_top + 76, card_width, 92))
        if view_model.has_active_session:
            button("resume_session", "RESUME CURRENT SESSION", pygame.Rect(content_left, content_top + 170, 240, 30))
        if view_model.recent_sessions:
            label("RECENT SESSIONS", (content_left, content_top + 204), theme.MUTED_TEXT, fonts[2])
            for index, recent in enumerate(view_model.recent_sessions[:5]):
                text = recent.display_name
                if recent.mode.value == "upload":
                    button(
                        f"recent:{recent.id}", text,
                        pygame.Rect(content_left, content_top + 234 + index * 38, content_width, 32),
                        disabled=view_model.has_active_session,
                    )
                else:
                    button(
                        f"recent:{recent.id}", f"{text}  ·  LIVE",
                        pygame.Rect(content_left, content_top + 234 + index * 38, content_width, 32),
                        disabled=True,
                    )
    elif screen == "upload_select":
        label("Choose an audio file", (content_left, content_top), theme.TEXT, fonts[0])
        label("WAV  /  MP3  /  FLAC     Up to 10 minutes", (content_left, content_top + 48), theme.MUTED_TEXT)
        for index, voice in enumerate(("3rd", "5th", "octave")):
            selected = voice in upload_voices
            button(f"upload_voice:{voice}", f"{'ON' if selected else 'OFF'}  {voice.upper()}", pygame.Rect(content_left + index * 142, content_top + 94, 128, 42))
        button("browse", "BROWSE FILE", pygame.Rect(content_left, content_top + 164, 210, 52), primary=True)
        label("Drop a supported audio file into this window", (content_left + 230, content_top + 180), theme.MUTED_TEXT)
        label(upload_path_entry or "Type a file path, then choose Browse", (content_left, content_top + 228), theme.TEXT if upload_path_entry else theme.MUTED_TEXT)
        if upload_start_error:
            label(f"Could not start upload: {upload_start_error}", (content_left, content_top + 260), theme.ERROR)
        if session and session.upload_error_message:
            label(session.upload_error_message, (content_left, content_top + 276), theme.ERROR)
    elif screen == "upload_analyzing":
        label("Analyzing your recording", (content_left, content_top), theme.TEXT, fonts[0])
        if session:
            phase = (session.upload_phase or "Preparing").replace("_", " ").title()
            label(phase, (content_left, content_top + 64), theme.ACCENT)
            total = session.upload_total_units
            completed = session.upload_completed_units
            progress_text = upload_progress_label(completed, total)
            label(progress_text, (content_left, content_top + 98), theme.MUTED_TEXT)
            if total is not None and total > 0 and completed <= total:
                track = pygame.Rect(content_left, content_top + 132, min(content_width, 560), 12)
                pygame.draw.rect(surface, theme.BORDER, track, border_radius=6)
                fill_width = round(track.width * completed / total)
                if fill_width:
                    pygame.draw.rect(surface, theme.ACCENT, pygame.Rect(track.left, track.top, fill_width, track.height), border_radius=6)
            if session.upload_error_message:
                label(f"{session.upload_error_code or 'Error'}: {session.upload_error_message}", (content_left, content_top + 168), theme.ERROR)
            if session.upload_status in {"failed", "cancelled"}:
                button("retry", "TRY AGAIN", pygame.Rect(content_left, content_top + 212, 150, 48), primary=True)
            elif session.upload_status == "completed":
                button("open_configure", "CONFIGURE", pygame.Rect(content_left, content_top + 212, 150, 48), primary=True)
            elif session.upload_status == "cancelling":
                label("Stopping analysis...", (content_left, content_top + 188), theme.WARNING)
            else:
                button("cancel_upload", "CANCEL", pygame.Rect(content_left, content_top + 212, 150, 48))
    elif screen == "upload_configure":
        label("Configure harmonies", (content_left, content_top), theme.TEXT, fonts[0])
        if session:
            source = session.source_file or "Uploaded recording"
            label(source, (content_left, content_top + 54), theme.MUTED_TEXT)
            key = "Auto-detected key unavailable" if session.key_root is None else f"{session.key_root} {session.key_scale_type or ''}".strip()
            detected = "unavailable" if session.detected_key_root is None else f"{session.detected_key_root} {session.detected_key_scale_type or ''}".strip()
            label(f"Detected  ·  {detected}    Effective  ·  {key}", (content_left, content_top + 92), theme.ACCENT)
            if session.upload_status == "running":
                target = "selected key" if session.pending_key_root is None else f"{session.pending_key_root} {session.pending_key_scale_type or ''}".strip()
                label(f"Rendering harmonies for {target}; playback is unavailable until matching buffers are ready.", (content_left, content_top + 112), theme.WARNING, fonts[2])
            elif session.upload_status == "failed":
                label("Harmony rendering failed. Choose a key to retry.", (content_left, content_top + 112), theme.ERROR, fonts[2])
            elif session.upload_status == "cancelled":
                label("Harmony rendering was cancelled. Choose a key to retry.", (content_left, content_top + 112), theme.WARNING, fonts[2])
            button("key:root:prev", "KEY −", pygame.Rect(content_left, content_top + 128, 96, 38), disabled=session_end_pending)
            button("key:root:next", "KEY +", pygame.Rect(content_left + 104, content_top + 128, 96, 38), disabled=session_end_pending)
            button("key:scale", "TOGGLE MAJOR/MINOR", pygame.Rect(content_left + 208, content_top + 128, 190, 38), disabled=session_end_pending)
            label("Rendered voices", (content_left, content_top + 174), theme.MUTED_TEXT)
            for index, voice in enumerate(session.available_voices):
                voice_enabled = dict(session.voices).get(voice, False)
                button(f"voice:{voice}", f"{'ON' if voice_enabled else 'OFF'}  {voice.upper()}", pygame.Rect(content_left + index * 150, content_top + 200, 136, 46), disabled=session_end_pending)
            button(
                "open_playback", "OPEN PLAYBACK", pygame.Rect(content_left, content_top + 274, 190, 50),
                primary=True, disabled=session.upload_status != "completed" or session_end_pending,
            )
            if not session_end_pending:
                button("end_session", "END SESSION", pygame.Rect(content_left + 204, content_top + 274, 190, 50))
    elif screen == "upload_playback":
        label("Synchronized playback", (content_left, content_top), theme.TEXT, fonts[0])
        if session:
            label(session.source_file or "Uploaded recording", (content_left, content_top + 51), theme.MUTED_TEXT)
            if view_model.waveform:
                wave_rect = pygame.Rect(content_left, content_top + 88, content_width, 96)
                pygame.draw.rect(surface, theme.BACKGROUND, wave_rect, border_radius=8)
                midpoint = wave_rect.centery
                for index, (low, high) in enumerate(view_model.waveform):
                    x = wave_rect.left + index * wave_rect.width // len(view_model.waveform)
                    y_low = midpoint - int(high * wave_rect.height * 0.43)
                    y_high = midpoint - int(low * wave_rect.height * 0.43)
                    pygame.draw.line(surface, theme.ACCENT, (x, y_low), (x, y_high), 1)
            else:
                label("Original waveform unavailable", (content_left, content_top + 115), theme.MUTED_TEXT)
            duration = max(session.playback_duration_seconds, 0.0)
            position = min(max(session.playback_position_seconds, 0.0), duration) if duration else 0.0
            seek = pygame.Rect(content_left, content_top + 198, content_width, 16)
            pygame.draw.rect(surface, theme.BORDER, seek, border_radius=8)
            if duration > 0:
                played_width = round(seek.width * position / duration)
                if played_width:
                    pygame.draw.rect(surface, theme.PRIMARY, pygame.Rect(seek.left, seek.top, played_width, seek.height), border_radius=8)
            controls["seek"] = seek
            label(f"{_format_time(position)}  /  {_format_time(duration)}", (content_left, content_top + 222), theme.MUTED_TEXT)
            play_name = "play_again" if session.playback_ended else ("pause" if session.playback_is_playing else "play")
            play_text = "PLAY AGAIN" if session.playback_ended else ("PAUSE" if session.playback_is_playing else "PLAY")
            button(play_name, play_text, pygame.Rect(content_left, content_top + 260, 150, 48), primary=True, disabled=duration <= 0)
            button("nav_home", "HOME", pygame.Rect(content_left + 164, content_top + 260, 112, 48))
            if not session_end_pending:
                button("end_session", "END SESSION", pygame.Rect(content_left + 292, content_top + 260, 160, 48))
            for index, voice in enumerate(session.available_voices):
                voice_enabled = dict(session.voices).get(voice, False)
                button(f"voice:{voice}", f"{'ON' if voice_enabled else 'OFF'}  {voice.upper()}", pygame.Rect(content_left + index * 142, content_top + 328, 128, 44), disabled=session_end_pending)
            if session.playback_error_message:
                label(f"Playback issue: {session.playback_error_message}", (content_left, content_top + 386), theme.ERROR)
            label("Mix control unavailable", (content_left, content_top + 418), theme.MUTED_TEXT)
    elif screen in {"live_configure", "live_session"}:
        if screen == "live_configure":
            label("Live Configure", (content_left, content_top), theme.TEXT, fonts[0])
            has_mic = bool(input_devices)
            if not has_mic:
                label("No microphone connected. Connect a microphone to start singing.", (content_left, content_top + 58), theme.WARNING)
            else:
                label(f"Microphone ready: {input_device_selector or 'System default'}", (content_left, content_top + 58), theme.SUCCESS)
            if audio_device_error:
                label(f"Audio device issue: {audio_device_error}", (content_left, content_top + 92), theme.ERROR)
            if live_start_error:
                label(f"Could not start: {live_start_error}", (content_left, content_top + 126), theme.ERROR)
            # Key selector row
            label("Key", (content_left, content_top + 155), theme.MUTED_TEXT)
            if live_key is not None:
                key_display = f"{live_key.root}  {live_key.scale_type.upper()}"
            else:
                key_display = "AUTO"
            label(key_display, (content_left + 52, content_top + 153), theme.ACCENT)
            button("live_key:root:prev", "KEY −", pygame.Rect(content_left + 148, content_top + 145, 86, 32))
            button("live_key:root:next", "KEY +", pygame.Rect(content_left + 242, content_top + 145, 86, 32))
            button("live_key:scale", "MAJ/MIN", pygame.Rect(content_left + 336, content_top + 145, 100, 32),
                   disabled=live_key is None)
            button("live_key:auto", "AUTO", pygame.Rect(content_left + 444, content_top + 145, 76, 32),
                   primary=live_key is None)
            button(
                "start_live",
                "START SINGING",
                pygame.Rect(content_left, content_top + 196, 190, 52),
                disabled=not has_mic or bool(audio_device_error) or session_end_pending,
                primary=True,
            )
        else:
            label("Live Session", (content_left, content_top), theme.TEXT, fonts[0])
            if session:
                in_tune_text = "IN TUNE" if session.in_tune else (
                    "TRACKING" if session.tracking_state == "tracking" else (
                        "TRACKING LOST" if session.tracking_state == "tracking_lost" else "LISTENING..."
                    )
                )
                in_tune_color = theme.SUCCESS if session.in_tune else (
                    theme.ACCENT if session.tracking_state == "tracking" else (
                        theme.WARNING if session.tracking_state == "tracking_lost" else theme.MUTED_TEXT
                    )
                )
                label(in_tune_text, (content_left, content_top + 48), in_tune_color)
                f0_text = f"Pitch: {session.current_f0:.1f} Hz (Confidence: {session.tracking_confidence:.0%})" if session.current_f0 else "Pitch: Listening..."
                label(f0_text, (content_left, content_top + 76), theme.TEXT)
                det_key = f"{session.detected_key_root} {session.detected_key_scale_type or ''}".strip() if session.detected_key_root else "detecting..."
                eff_key = f"{session.key_root} {session.key_scale_type or ''}".strip() if session.key_root else "auto"
                label(f"Key: {eff_key}  (detected: {det_key})", (content_left, content_top + 104), theme.ACCENT)
                track = pygame.Rect(content_left, content_top + 134, min(content_width, 400), 8)
                pygame.draw.rect(surface, theme.BORDER, track, border_radius=4)
                level_fill = round(track.width * min(1.0, max(0.0, session.input_level * 4.0)))
                if level_fill:
                    pygame.draw.rect(surface, theme.PRIMARY, pygame.Rect(track.left, track.top, level_fill, track.height), border_radius=4)
                if session.physical_latency_ms is not None:
                    lat_text = f"Latency: {session.physical_latency_ms:.1f} ms ({session.latency_provenance})"
                elif session.pipeline_latency_ms is not None:
                    lat_text = f"Pipeline latency: {session.pipeline_latency_ms:.1f} ms ({session.latency_provenance})"
                else:
                    lat_text = "Latency: calculating..."
                label(lat_text, (content_left, content_top + 152), theme.MUTED_TEXT, fonts[2])
                if session.audio_error_message:
                    label(session.audio_error_message, (content_left, content_top + 172), theme.ERROR, fonts[2])
            else:
                label("No live session active", (content_left, content_top + 58), theme.MUTED_TEXT)
            button("stop_live", "STOP", pygame.Rect(content_left, content_top + 196, 140, 52), primary=True, disabled=session_end_pending)
            if view_model.has_active_session and not session_end_pending:
                button("end_session", "END SESSION", pygame.Rect(content_left + 156, content_top + 196, 160, 52))
    else:
        title = "Review & Export" if screen == "review_export" else "Settings"
        label(title, (content_left, content_top), theme.TEXT, fonts[0])
        if screen == "review_export":
            if session is None or not session.has_recorded_audio:
                label("No completed recording is available to export.", (content_left, content_top + 58), theme.WARNING)
            else:
                label("Include harmony voices in the WAV", (content_left, content_top + 58), theme.MUTED_TEXT)
                selected = set(view_model.export_selected_voices)
                for index, voice in enumerate(session.available_voices):
                    button(
                        f"export_voice:{voice}",
                        f"{'ON' if voice in selected else 'OFF'}  {voice.upper()}",
                        pygame.Rect(content_left + index * 150, content_top + 92, 138, 42),
                        disabled=view_model.export_in_progress,
                        primary=voice in selected,
                    )
                label("Output format: WAV", (content_left, content_top + 148), theme.MUTED_TEXT)
                if view_model.export_in_progress:
                    phase = "Writing WAV…" if view_model.export_phase == "writing" else "Rendering WAV…"
                    label(f"{phase}  {view_model.export_progress_pct:.1f}%" if view_model.export_progress_pct is not None else phase,
                          (content_left, content_top + 184), theme.ACCENT)
                    track = pygame.Rect(content_left, content_top + 215, min(content_width, 560), 14)
                    pygame.draw.rect(surface, theme.BORDER, track, border_radius=7)
                    if view_model.export_progress_pct is not None:
                        fill = round(track.width * view_model.export_progress_pct / 100.0)
                        if fill:
                            pygame.draw.rect(surface, theme.ACCENT, pygame.Rect(track.left, track.top, fill, track.height), border_radius=7)
                    button("export_cancel", "CANCEL EXPORT", pygame.Rect(content_left, content_top + 246, 180, 44))
                else:
                    if view_model.export_error or export_ui_error:
                        label(f"Export failed; session retained for retry: {view_model.export_error or export_ui_error}",
                              (content_left, content_top + 182), theme.ERROR, fonts[2])
                    elif view_model.export_status == "cancelled":
                        label("Export cancelled. Your session and selection are retained.",
                              (content_left, content_top + 182), theme.WARNING, fonts[2])
                    button("export_start", "EXPORT WAV" if view_model.export_status not in {"failed", "cancelled"} else "RETRY EXPORT",
                           pygame.Rect(content_left, content_top + 226, 190, 48), primary=True)
                    button("discard_session", "DISCARD SESSION", pygame.Rect(content_left + 210, content_top + 226, 190, 48))
                if discard_confirmation_pending:
                    label("Discard this session and its recorded buffers?", (content_left, content_top + 292), theme.WARNING)
                    button("discard_confirm", "CONFIRM DISCARD", pygame.Rect(content_left, content_top + 326, 190, 42), primary=True)
                    button("discard_cancel", "KEEP SESSION", pygame.Rect(content_left + 204, content_top + 326, 160, 42))
            if export_ui_error and not (session and session.has_recorded_audio):
                label(export_ui_error, (content_left, content_top + 92), theme.ERROR, fonts[2])
        else:
            disabled = settings_retry_pending
            label("Default key behavior", (content_left, content_top), theme.MUTED_TEXT)
            button("setting:key:auto", "AUTO", pygame.Rect(content_left + 205, content_top - 4, 130, 36), disabled=disabled, primary=view_model.default_key_behavior == "auto")
            button("setting:key:last_used", "LAST USED", pygame.Rect(content_left + 345, content_top - 4, 150, 36), disabled=disabled, primary=view_model.default_key_behavior == "last_used")
            label("Default style", (content_left, content_top + 58), theme.MUTED_TEXT)
            for index, (value, text) in enumerate((("pop", "POP"), ("choir", "CHOIR"), ("barbershop", "BARBERSHOP"))):
                button(
                    f"setting:style:{value}", text,
                    pygame.Rect(content_left + 205 + index * 142, content_top + 52, 132, 36),
                    disabled=disabled or value == "barbershop",
                    primary=view_model.default_style_preset == value,
                )
            label("Choir uses fixed octave spacing. Barbershop is disabled until its mapping is approved.", (content_left, content_top + 88), theme.WARNING, fonts[2])
            label("Latency frame size", (content_left, content_top + 116), theme.MUTED_TEXT)
            button("setting:frame:down", "- 256", pygame.Rect(content_left + 205, content_top + 110, 112, 36), disabled=disabled or view_model.latency_buffer_frame_size <= 256)
            label(f"{view_model.latency_buffer_frame_size} samples", (content_left + 332, content_top + 118), theme.TEXT)
            button("setting:frame:up", "+ 256", pygame.Rect(content_left + 475, content_top + 110, 112, 36), disabled=disabled or view_model.latency_buffer_frame_size >= 4096)
            label("Smaller frames can reduce buffering but provide less pitch context; larger frames can improve tracking stability and may increase delay.", (content_left, content_top + 155), theme.MUTED_TEXT, fonts[2])
            label("This preference does not change Live sessions in this phase.", (content_left, content_top + 180), theme.MUTED_TEXT, fonts[2])
            input_label = input_device_selector or "System default"
            output_label = output_device_selector or "System default"
            label(f"Input: {input_label[:34]}", (content_left, content_top + 218), theme.MUTED_TEXT, fonts[2])
            button("setting:input_device:previous", "PREV", pygame.Rect(content_left + 360, content_top + 210, 68, 32), disabled=disabled or not input_devices)
            button("setting:input_device:next", "NEXT", pygame.Rect(content_left + 434, content_top + 210, 68, 32), disabled=disabled or not input_devices)
            button("setting:input_device:default", "SYSTEM DEFAULT", pygame.Rect(content_left + 508, content_top + 210, 150, 32), disabled=disabled)
            label(f"Output: {output_label[:32]}", (content_left, content_top + 254), theme.MUTED_TEXT, fonts[2])
            button("setting:output_device:previous", "PREV", pygame.Rect(content_left + 360, content_top + 246, 68, 32), disabled=disabled or not output_devices)
            button("setting:output_device:next", "NEXT", pygame.Rect(content_left + 434, content_top + 246, 68, 32), disabled=disabled or not output_devices)
            button("setting:output_device:default", "SYSTEM DEFAULT", pygame.Rect(content_left + 508, content_top + 246, 150, 32), disabled=disabled)
            button("refresh_audio_devices", "REFRESH DEVICES", pygame.Rect(content_left + 670, content_top + 228, 150, 34), disabled=disabled)
            if audio_device_error:
                label(f"Device selection issue: {audio_device_error}", (content_left, content_top + 284), theme.ERROR, fonts[2])
            elif not audio_devices_loaded:
                label("Refresh devices to enumerate available input and output devices.", (content_left, content_top + 284), theme.MUTED_TEXT, fonts[2])
            button(
                "setting:reduced_motion",
                f"REDUCED MOTION: {'ON' if view_model.reduced_motion else 'OFF'}",
                pygame.Rect(content_left, content_top + 310, 260, 36),
                disabled=disabled,
            )
            label("Visualizer motion behavior is not implemented yet.", (content_left + 280, content_top + 319), theme.MUTED_TEXT, fonts[2])
            if settings_warning:
                label("Warning: settings could not be read; defaults are in use.", (content_left, content_top + 352), theme.WARNING, fonts[2])
                label("Close Harmonix and check the settings file or its permissions.", (content_left, content_top + 374), theme.WARNING, fonts[2])
            if settings_error:
                label(f"Could not save setting: {settings_error}", (content_left, content_top + 352), theme.ERROR, fonts[2])
            if settings_retry_pending:
                button("settings_retry", "RETRY SAVE", pygame.Rect(content_left, content_top + 370, 160, 44), primary=True)
            button("nav_home", "HOME", pygame.Rect(panel.left + 154, panel.bottom - 54, 96, 34))

    if close_confirmation_pending:
        shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        surface.blit(shade, (0, 0))
        dialog = pygame.Rect(width // 2 - 260, height // 2 - 105, 520, 210)
        pygame.draw.rect(surface, theme.SURFACE, dialog, border_radius=12)
        pygame.draw.rect(surface, theme.WARNING, dialog, width=2, border_radius=12)
        label("An export is running. Closing will cancel it.", (dialog.left + 24, dialog.top + 32), theme.TEXT)
        label("Wait for the worker to stop before application cleanup.", (dialog.left + 24, dialog.top + 66), theme.MUTED_TEXT, fonts[2])
        button("close_export_confirm", "CANCEL EXPORT & CLOSE", pygame.Rect(dialog.left + 24, dialog.top + 118, 238, 44), primary=True)
        button("close_export_cancel", "KEEP WORKING", pygame.Rect(dialog.left + 278, dialog.top + 118, 218, 44))
    elif export_toast:
        label(export_toast, (panel.left + 28, panel.bottom - 94), theme.ACCENT, fonts[2])
        if "saved" in export_toast.lower():
            button("show_export_folder", "SHOW IN FOLDER", pygame.Rect(panel.right - 192, panel.bottom - 104, 164, 36))
    button("nav_settings", "SETTINGS", pygame.Rect(panel.left + 28, panel.bottom - 54, 112, 34), disabled=close_confirmation_pending or discard_confirmation_pending)
    if close_confirmation_pending:
        enabled.clear()
        enabled.update({"close_export_confirm", "close_export_cancel"})
    elif discard_confirmation_pending:
        enabled.clear()
        enabled.update({"discard_confirm", "discard_cancel"})
    return ShellLayout(controls, enabled)


def _format_time(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    return f"{whole_seconds // 60}:{whole_seconds % 60:02d}"
