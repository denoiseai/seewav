import argparse
import json
import math
import subprocess as sp
import sys
import tempfile
from pathlib import Path
import cairo
import cupy as cp  # CuPy para cálculos en GPU
import numpy as np  # Necesario para interoperabilidad y formato final
import tqdm
import time

_is_main = False


def colorize(text, color):
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def fatal(msg):
    if _is_main:
        head = "error: "
        if sys.stderr.isatty():
            head = colorize("error: ", 1)
        print(head + str(msg), file=sys.stderr)
        sys.exit(1)


def read_info(media):
    proc = sp.run([
        'ffprobe', "-loglevel", "panic",
        str(media), '-print_format', 'json', '-show_format', '-show_streams'
    ], capture_output=True)
    if proc.returncode:
        raise IOError(f"{media} does not exist or is of a wrong type.")
    return json.loads(proc.stdout.decode('utf-8'))


def read_audio(audio, seek=None, duration=None):
    info = read_info(audio)
    stream = info['streams'][0]
    if stream["codec_type"] != "audio":
        raise ValueError(f"{audio} should contain only audio.")
    channels = stream['channels']
    samplerate = float(stream['sample_rate'])

    command = ['ffmpeg', '-y']
    command += ['-loglevel', 'panic']
    if seek is not None:
        command += ['-ss', str(seek)]
    command += ['-i', audio]
    if duration is not None:
        command += ['-t', str(duration)]
    command += ['-f', 'f32le']
    command += ['-']

    proc = sp.run(command, check=True, capture_output=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    return wav.reshape(-1, channels).T, samplerate


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


kernel_code_fusion = '''
extern "C" __global__
void compute_envelope_fusion(const float* __restrict__ wav, float* __restrict__ out, int n, int window, int stride) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < window; i++) {
        int pos = idx * stride + i;
        sum += max(0.0f, wav[pos]);
    }
    float envelope = 1.9f * (1.0f / (1.0f + expf(-2.5f * (sum / window))) - 0.5f);
    out[idx] = envelope;
}
'''

kernel_fusion = cp.RawKernel(kernel_code_fusion, 'compute_envelope_fusion')


def envelope_gpu_optimized(wav, window, stride):
    n = (wav.size - window) // stride + 1
    out = cp.zeros(n, dtype=cp.float32)
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    kernel_fusion((blocks,), (threads_per_block,), (wav, out, wav.size, window, stride))
    return out


def draw_env(envs, out, fg_colors, bg_color, size):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
    ctx = cairo.Context(surface)
    ctx.scale(*size)

    ctx.set_source_rgb(*bg_color)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()

    K = len(envs)
    T = len(envs[0])
    pad_ratio = 0.1
    width = 1. / (T * (1 + 2 * pad_ratio))
    pad = pad_ratio * width
    delta = 2 * pad + width

    ctx.set_line_width(width)
    for step in range(T):
        for i in range(K):
            half = 0.5 * envs[i][step]
            half /= K
            midrule = (1 + 2 * i) / (2 * K)
            ctx.set_source_rgb(*fg_colors[i])
            ctx.move_to(pad + step * delta, midrule - half)
            ctx.line_to(pad + step * delta, midrule)
            ctx.stroke()
            ctx.set_source_rgba(*fg_colors[i], 0.8)
            ctx.move_to(pad + step * delta, midrule)
            ctx.line_to(pad + step * delta, midrule + 0.9 * half)
            ctx.stroke()

    surface.write_to_png(out)


def interpole(x1, y1, x2, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def visualize(audio,
              tmp,
              out,
              seek=None,
              duration=None,
              rate=30,
              bars=30,
              speed=4,
              time=0.4,
              oversample=3,
              fg_color=(.2, .2, .2),
              fg_color2=(.5, .3, .6),
              bg_color=(1, 1, 1),
              size=(400, 400),
              stereo=False,
              ):
    try:
        wav, sr = read_audio(audio, seek=seek, duration=duration)
    except (IOError, ValueError) as err:
        fatal(err)
        raise

    wavs = []
    if stereo:
        assert wav.shape[0] == 2, 'Stereo requires stereo input audio.'
        wavs.append(wav[0])
        wavs.append(wav[1])
    else:
        wav = wav.mean(0)
        wavs.append(wav)

    for i, wav in enumerate(wavs):
        wavs[i] = wav / wav.std()

    window = int(sr * time / bars)
    stride = int(window / oversample)

    envs = []
    for wav in wavs:
        env = envelope_gpu_optimized(cp.array(wav), window, stride)
        env = cp.pad(env, (0, 2 * bars))  # Ajustar padding
        envs.append(cp.asnumpy(env))  # Convertimos a NumPy para Cairo

    duration = len(wavs[0]) / sr
    frames = int(rate * duration)

    print("Generating the frames...")
    smooth = cp.hanning(bars)
    for idx in tqdm.tqdm(range(frames), unit=" frames", ncols=80):
        pos = idx * bars
        off = int(pos)
        denvs = []
        for env in envs:
            if off + 1 + bars > len(env):
                break

            env1 = cp.array(env[off:off + bars])
            env2 = cp.array(env[off + 1:off + 1 + bars])

            maxvol = cp.log10(1e-4 + env2.max()) * 10
            maxvol = maxvol.item()  # Convertir a escalar de Python

            # Reemplaza cp.clip por max y min
            speedup = max(0.5, min(2, interpole(-6, 0.5, 0, 2, maxvol)))

            loc = 0
            w = sigmoid(speed * speedup * (loc - 0.5))

            denv = (1 - w) * env1 + w * env2
            denv *= smooth
            denvs.append(cp.asnumpy(denv))

        if not denvs:
            continue

        draw_env(denvs, tmp / f"{idx:06d}.png", (fg_color, fg_color2), bg_color, size)


    audio_cmd = []
    if seek is not None:
        audio_cmd += ["-ss", str(seek)]
    audio_cmd += ["-i", audio.resolve()]
    if duration is not None:
        audio_cmd += ["-t", str(duration)]
    print("Encoding the animation video... ")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "panic", "-r",
        str(rate), "-f", "image2", "-s", f"{size[0]}x{size[1]}", "-i", "%06d.png"
    ] + audio_cmd + [
        "-c:a", "aac", "-vcodec", "h264_nvenc", "-crf", "10", "-pix_fmt", "yuv420p",
        out.resolve()
    ],
           check=True,
           cwd=tmp)


def parse_color(colorstr):
    try:
        r, g, b = [float(i) for i in colorstr.split(",")]
        return r, g, b
    except ValueError:
        fatal("Format for color is 3 floats separated by commas 0.xx,0.xx,0.xx, rgb order")
        raise


def main():
    start_time = time.time()  # Inicio de medición de tiempo
    parser = argparse.ArgumentParser(
        'seewav', description="Generate a nice mp4 animation from an audio file.")
    parser.add_argument("-r", "--rate", type=int, default=60, help="Video framerate.")
    parser.add_argument("--stereo", action='store_true',
                        help="Create 2 waveforms for stereo files.")
    parser.add_argument("-c",
                        "--color",
                        default=[0.03, 0.6, 0.3],
                        type=parse_color,
                        dest="color",
                        help="Color of the bars as `r,g,b` in [0, 1].")
    parser.add_argument("-c2",
                        "--color2",
                        default=[0.5, 0.3, 0.6],
                        type=parse_color,
                        dest="color2",
                        help="Color of the second waveform as `r,g,b` in [0, 1] (for stereo).")
    parser.add_argument("--white", action="store_true",
                        help="Use white background. Default is black.")
    parser.add_argument("-B",
                        "--bars",
                        type=int,
                        default=50,
                        help="Number of bars on the video at once")
    parser.add_argument("-O", "--oversample", type=float, default=4,
                        help="Lower values will feel less reactive.")
    parser.add_argument("-T", "--time", type=float, default=0.4,
                        help="Amount of audio shown at once on a frame.")
    parser.add_argument("-S", "--speed", type=float, default=4,
                        help="Higher values means faster transitions between frames.")
    parser.add_argument("-W",
                        "--width",
                        type=int,
                        default=480,
                        help="width in pixels of the animation")
    parser.add_argument("-H",
                        "--height",
                        type=int,
                        default=300,
                        help="height in pixels of the animation")
    parser.add_argument("-s", "--seek", type=float, help="Seek to time in seconds in video.")
    parser.add_argument("-d", "--duration", type=float, help="Duration in seconds from seek time.")
    parser.add_argument("audio", type=Path, help='Path to audio file')
    parser.add_argument("out",
                        type=Path,
                        nargs='?',
                        default=Path('out.mp4'),
                        help='Path to output file. Default is ./out.mp4')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        visualize(args.audio,
                  Path(tmp),
                  args.out,
                  seek=args.seek,
                  duration=args.duration,
                  rate=args.rate,
                  bars=args.bars,
                  speed=args.speed,
                  oversample=args.oversample,
                  time=args.time,
                  fg_color=args.color,
                  fg_color2=args.color2,
                  bg_color=[1. * bool(args.white)] * 3,
                  size=(args.width, args.height),
                  stereo=args.stereo)
    end_time = time.time()  # Fin de medición de tiempo
    print(f"Execution time: {end_time - start_time:.2f} seconds")  # Imprimir tiempo total


if __name__ == "__main__":
    _is_main = True
    main()
