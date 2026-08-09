const exercises = [
  { sentence: "I really like this project.", targets: [1, 4] },
  { sentence: "Please bring the blue folder.", targets: [3, 4] },
  { sentence: "She finished the report today.", targets: [1, 3, 4] },
  { sentence: "Could you send me the details?", targets: [1, 2, 5] },
  { sentence: "We need to solve this problem.", targets: [1, 3, 5] },
  { sentence: "The meeting starts at nine o'clock.", targets: [1, 2, 4, 5] },
  { sentence: "I bought a new computer yesterday.", targets: [1, 3, 4, 5] },
  { sentence: "They decided to visit Taiwan.", targets: [1, 3, 4] },
  { sentence: "Can you explain that again?", targets: [1, 2, 4] },
  { sentence: "My teacher gave us helpful feedback.", targets: [1, 2, 4, 5] }
];

let selectedAudio = null;
let recorder = null;
let activeExercise = null;
let recordingTimer = null;
let secondsRecorded = 0;

const fileInput = document.querySelector("#audio-file");
const uploadZone = document.querySelector("#upload-zone");
const fileName = document.querySelector("#file-name");
const preview = document.querySelector("#audio-preview");
const recordButton = document.querySelector("#record-button");
const recordingTime = document.querySelector("#recording-time");
const analyzeButton = document.querySelector("#analyze-button");
const analysisError = document.querySelector("#analysis-error");
const stressThreshold = document.querySelector("#stress-threshold");
const thresholdValue = document.querySelector("#threshold-value");

function formatTime(seconds) {
  return `00:${String(seconds).padStart(2, "0")}`;
}

function setSelectedAudio(file) {
  selectedAudio = file;
  fileName.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
}

function setRecordingButton(button, recording) {
  button.classList.toggle("is-recording", recording);
  button.textContent = recording ? "■ 停止錄音" : "● 開始錄音";
}

async function startRecording(button, exerciseIndex = null) {
  try {
    recorder = new WavRecorder();
    await recorder.start();

    activeExercise = exerciseIndex;
    secondsRecorded = 0;
    setRecordingButton(button, true);

    recordingTimer = window.setInterval(() => {
      secondsRecorded += 1;

      if (exerciseIndex === null) {
        recordingTime.textContent = formatTime(secondsRecorded);
      }
    }, 1000);
  } catch (error) {
    alert("無法使用麥克風。請允許瀏覽器存取麥克風後再試一次。");
    console.error(error);
  }
}

async function stopRecording(button) {
  window.clearInterval(recordingTimer);

  const blob = await recorder.stop();
  const file = new File([blob], "recording.wav", {
    type: "audio/wav"
  });

  const exerciseIndex = activeExercise;

  recorder = null;
  activeExercise = null;
  setRecordingButton(button, false);

  if (exerciseIndex === null) {
    setSelectedAudio(file);
    recordingTime.textContent = formatTime(secondsRecorded);
    return;
  }

  await analyzeExercise(exerciseIndex, file, button);
}

async function predict(file, transcription = "", threshold = 0.70) {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("sentence_stress_threshold", threshold);

  if (transcription.trim()) {
    formData.append("transcription", transcription.trim());
  }

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.detail || "分析失敗。");
  }

  return result;
}

function createWordCard(item, showPhones) {
  const card = document.createElement("article");
  card.className = `word-card${item.sentence_stress ? " stressed" : ""}`;

  const word = document.createElement("div");
  word.className = "word-name";
  word.textContent = item.word;

  const pos = document.createElement("div");
  pos.className = "pos";
  pos.textContent = `詞性：${item.pos}`;

  card.append(word, pos);

  if (showPhones) {
    const phones = document.createElement("div");
    phones.className = "phone-list";

    item.phones.forEach((phoneDetail) => {
      const phone = document.createElement("span");
      phone.className = `phone${phoneDetail.stress ? " stressed" : ""}`;
      phone.textContent = phoneDetail.phone;
      phones.append(phone);
    });

    card.append(phones);
  }

  return card;
}

function renderAnalysis(result) {
  const transcript = document.querySelector("#analysis-transcript");
  const wordGrid = document.querySelector("#analysis-words");
  const phoneStatus = document.querySelector("#phone-status");

  transcript.replaceChildren();
  wordGrid.replaceChildren();

  result.words.forEach((item) => {
    const word = document.createElement("span");
    word.className = `transcript-word${item.sentence_stress ? " stressed" : ""}`;
    word.textContent = item.word;

    transcript.append(word);
    wordGrid.append(createWordCard(item, result.phone_stress_available));
  });

  phoneStatus.className = `status${
    result.phone_stress_available ? " available" : ""
  }`;

  phoneStatus.textContent = result.phone_stress_available
    ? "音素重音：已分析"
    : "音素重音：需要逐字稿";

  document.querySelector("#analysis-result").hidden = false;
}

async function analyzeExercise(index, file, button) {
  const message = document.querySelector(`#exercise-message-${index}`);

  button.disabled = true;
  button.textContent = "分析中…";
  message.hidden = true;

  try {
    const result = await predict(file, exercises[index].sentence);
    renderPracticeResult(index, result);
  } catch (error) {
    message.textContent = `分析失敗：${error.message}`;
    message.hidden = false;
  } finally {
    button.disabled = false;
    setRecordingButton(button, false);
  }
}

function renderPracticeResult(index, result) {
  const exercise = exercises[index];
  const container = document.querySelector(`#exercise-result-${index}`);

  container.replaceChildren();

  const wordFeedback = document.createElement("div");
  wordFeedback.className = "practice-word-feedback";

  const phoneFeedback = document.createElement("div");
  phoneFeedback.className = "practice-phone-feedback";

  result.words.forEach((item, wordIndex) => {
    const shouldStress = exercise.targets.includes(wordIndex);
    const predictedStress = Boolean(item.sentence_stress);

    let status = "neutral";

    if (shouldStress && predictedStress) {
      status = "correct";
    } else if (shouldStress && !predictedStress) {
      status = "missed";
    } else if (!shouldStress && predictedStress) {
      status = "extra";
    }

    const word = document.createElement("span");
    word.className = `practice-word ${status}`;
    word.textContent = item.word;
    wordFeedback.append(word);

    const phoneCard = document.createElement("div");
    phoneCard.className = `practice-phone-card ${status}`;

    const title = document.createElement("strong");
    title.textContent = item.word;

    const phones = document.createElement("div");
    phones.className = "practice-phone-list";

    item.phones.forEach((phoneDetail) => {
      const phone = document.createElement("span");

      phone.className = `practice-phone${
        phoneDetail.stress ? " stressed" : ""
      }`;

      phone.textContent = phoneDetail.phone;
      phones.append(phone);
    });

    phoneCard.append(title, phones);
    phoneFeedback.append(phoneCard);
  });

  const note = document.createElement("small");
  note.textContent =
    "單字顏色為句子重音回饋；黃色音素代表 WhiStressPhn 判定該音素具有重音。";

  container.append(wordFeedback, phoneFeedback, note);
  container.hidden = false;
}

function renderExercises() {
  const list = document.querySelector("#exercise-list");
  const template = document.querySelector("#exercise-template");

  exercises.forEach((exercise, index) => {
    const fragment = template.content.cloneNode(true);
    const words = exercise.sentence.replace(/[.?!]/g, "").split(" ");

    fragment.querySelector(".exercise-number").textContent =
      `練習 ${index + 1}`;

    fragment.querySelector(".exercise-sentence").textContent =
      exercise.sentence;

    fragment.querySelector(".exercise-target").innerHTML =
      `目標重音：<strong>${
        exercise.targets.map((target) => words[target]).join("、")
      }</strong>`;

    const button = fragment.querySelector(".exercise-record");
    const message = fragment.querySelector(".exercise-message");
    const result = fragment.querySelector(".exercise-result");

    
    message.id = `exercise-message-${index}`;
    result.id = `exercise-result-${index}`;

    button.addEventListener("click", async () => {
      if (recorder) {
        if (activeExercise === index) {
          await stopRecording(button);
        }
        return;
      }

      await startRecording(button, index);
    });

    list.append(fragment);
  });
}

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) {
    setSelectedAudio(fileInput.files[0]);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("dragging");
  });
});

uploadZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];

  if (file) {
    setSelectedAudio(file);
  }
});

recordButton.addEventListener("click", async () => {
  if (recorder) {
    if (activeExercise === null) {
      await stopRecording(recordButton);
    }
    return;
  }

  await startRecording(recordButton);
});

analyzeButton.addEventListener("click", async () => {
  analysisError.hidden = true;

  if (!selectedAudio) {
    analysisError.textContent = "請先上傳音檔或錄音。";
    analysisError.hidden = false;
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "分析中…";

  try {
    const result = await predict(
    selectedAudio,
    document.querySelector("#transcription").value,
    Number(stressThreshold.value)
  );

    renderAnalysis(result);
  } catch (error) {
    analysisError.textContent = `分析失敗：${error.message}`;
    analysisError.hidden = false;
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "開始分析";
  }
});

stressThreshold.addEventListener("input", () => {
  thresholdValue.textContent = Number(stressThreshold.value).toFixed(2);
});

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    const mode = button.dataset.mode;

    document.querySelectorAll(".mode-button").forEach((item) => {
      item.classList.toggle("is-active", item === button);
    });

    document.querySelector("#analysis-mode").hidden = mode !== "analysis";
    document.querySelector("#practice-mode").hidden = mode !== "practice";
  });
});

renderExercises();