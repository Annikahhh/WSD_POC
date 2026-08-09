class WavRecorder {
  constructor() {
    this.stream = null;
    this.context = null;
    this.source = null;
    this.processor = null;
    this.chunks = [];
    this.sampleRate = 44100;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: true
    });

    this.context = new AudioContext();

    await this.context.resume();

    if (this.context.state !== "running") {
      throw new Error("瀏覽器無法啟用錄音音訊環境。");
    }

    this.sampleRate = this.context.sampleRate;
    this.source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.chunks = [];

    this.processor.onaudioprocess = (event) => {
      const channelData = event.inputBuffer.getChannelData(0);

      if (channelData.length > 0) {
        this.chunks.push(new Float32Array(channelData));
      }
    };

    this.source.connect(this.processor);
    this.processor.connect(this.context.destination);
  }

  async stop() {
    if (!this.chunks.length) {
      throw new Error("沒有收到麥克風音訊。請確認瀏覽器麥克風權限。");
    }

    this.processor.disconnect();
    this.source.disconnect();

    this.stream.getTracks().forEach((track) => track.stop());

    await this.context.close();

    return new Blob([this.toWav()], {
      type: "audio/wav"
    });
  }

  toWav() {
    const length = this.chunks.reduce(
      (total, chunk) => total + chunk.length,
      0
    );

    const samples = new Float32Array(length);
    let offset = 0;

    this.chunks.forEach((chunk) => {
      samples.set(chunk, offset);
      offset += chunk.length;
    });

    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeText = (position, text) => {
      [...text].forEach((character, index) => {
        view.setUint8(position + index, character.charCodeAt(0));
      });
    };

    writeText(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    writeText(8, "WAVE");
    writeText(12, "fmt ");

    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, this.sampleRate, true);
    view.setUint32(28, this.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);

    writeText(36, "data");
    view.setUint32(40, samples.length * 2, true);

    samples.forEach((sample, index) => {
      const value = Math.max(-1, Math.min(1, sample));
      view.setInt16(44 + index * 2, value * 0x7fff, true);
    });

    return buffer;
  }
}