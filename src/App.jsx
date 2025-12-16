import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";

// --- SAFE GLOBAL VARIABLES ---
// We declare them here, but we don't touch them yet to prevent crashing.
let synth = null;
let utterance = null;
// -----------------------------

const ASL_SUGGESTIONS = ["HELLO", "YES", "NO", "THANK YOU", "I LOVE YOU", "HELP", "BATHROOM", "EAT"];

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const classifier = useRef(null);
  const handsRef = useRef(null); 
  
  const lastRunTime = useRef(0);

  const [isAppStarted, setIsAppStarted] = useState(false);
  const [activeTab, setActiveTab] = useState("train"); 
  const [status, setStatus] = useState("System Ready");
  const [currentGesture, setCurrentGesture] = useState("--");
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState([]);
  const [customLabel, setCustomLabel] = useState("");
  const [savedGestures, setSavedGestures] = useState([]); 
  const [prepTime, setPrepTime] = useState(2); 
  const [timerDisplay, setTimerDisplay] = useState(null); 
  const [facingMode, setFacingMode] = useState("user"); 

  const trainingLabelRef = useRef(null);
  const lastWordRef = useRef("");     
  const wordStableCount = useRef(0); 
  const actionCooldown = useRef(false); 
  const sentenceRef = useRef([]);

  useEffect(() => { sentenceRef.current = sentence; }, [sentence]);

  // --- INITIALIZE SPEECH SAFELY ---
  // We do this once when the app loads, so it doesn't crash the white screen.
  useEffect(() => {
    if ('speechSynthesis' in window) {
      synth = window.speechSynthesis;
      utterance = new SpeechSynthesisUtterance();
    }
  }, []);

  const initAI = async () => {
    try {
        setStatus("Loading Brain...");
        
        if (!window.tf || !window.knnClassifier || !window.Hands || !window.drawConnectors) {
            throw new Error("Internet Required for AI");
        }
        
        await window.tf.ready();
        classifier.current = window.knnClassifier.create();
        restoreBrain(); 

        setStatus("Starting Vision...");
        
        const hands = new window.Hands({ 
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` 
        });
        
        hands.setOptions({ 
            maxNumHands: 2, 
            modelComplexity: 1, 
            minDetectionConfidence: 0.5, 
            minTrackingConfidence: 0.5 
        });
        
        hands.onResults(onResults);
        handsRef.current = hands;
        
        requestAnimationFrame(processVideo);
        setStatus("System Ready");

    } catch (err) { alert("Error: " + err.message); setStatus("Retrying..."); setTimeout(initAI, 2000); }
  };

  const processVideo = async () => {
      const now = Date.now();
      if (now - lastRunTime.current > 30) {
          lastRunTime.current = now;

          if (
              webcamRef.current && 
              webcamRef.current.video && 
              webcamRef.current.video.readyState === 4
          ) {
              const video = webcamRef.current.video;
              const videoWidth = video.videoWidth;
              const videoHeight = video.videoHeight;

              if (canvasRef.current) {
                  canvasRef.current.width = videoWidth;
                  canvasRef.current.height = videoHeight;
              }

              if (handsRef.current) {
                  await handsRef.current.send({ image: video });
              }
          }
      }
      requestAnimationFrame(processVideo);
  };

  const switchCamera = () => {
      setFacingMode(prev => prev === "user" ? "environment" : "user");
      setStatus("Switching...");
  };

  const saveBrain = () => {
      if (!classifier.current) return;
      try {
          const dataset = classifier.current.getClassifierDataset();
          const datasetObj = {};
          Object.keys(dataset).forEach((key) => {
             datasetObj[key] = Array.from(dataset[key].dataSync());
          });
          localStorage.setItem("sign-speak-brain", JSON.stringify(datasetObj));
      } catch (e) {}
  };

  const restoreBrain = () => {
      try {
          const jsonStr = localStorage.getItem("sign-speak-brain");
          if (jsonStr) {
             const dataset = JSON.parse(jsonStr);
             const tensorObj = {};
             Object.keys(dataset).forEach((key) => {
                const numExamples = dataset[key].length / 126;
                tensorObj[key] = window.tf.tensor(dataset[key], [numExamples, 126]);
             });
             classifier.current.setClassifierDataset(tensorObj);
             setSavedGestures(Object.keys(dataset));
          }
      } catch (e) {}
  };

  const onResults = (results) => {
    if (!canvasRef.current) return;
    
    const ctx = canvasRef.current.getContext("2d");
    ctx.save();
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    if (results.multiHandLandmarks) {
      for (const landmarks of results.multiHandLandmarks) {
        if (window.drawConnectors) {
            window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
            window.drawLandmarks(ctx, landmarks, { color: '#FF0000', lineWidth: 2, radius: 5 });
        }
      }
    }
    ctx.restore();

    if (results.multiHandLandmarks.length > 0) {
        const values = new Array(126).fill(0);
        results.multiHandLandmarks.forEach((landmarks, handIndex) => {
            if (handIndex > 1) return;
            const wristX = landmarks[0].x; const wristY = landmarks[0].y; const wristZ = landmarks[0].z;
            for(let i=0; i<landmarks.length; i++){
                const offset = handIndex * 63; const base = offset + (i * 3);
                values[base] = landmarks[i].x - wristX;
                values[base + 1] = landmarks[i].y - wristY;
                values[base + 2] = landmarks[i].z - wristZ;
            }
        });
        const inputTensor = window.tf.tensor(values);
        if (trainingLabelRef.current) classifier.current.addExample(inputTensor, trainingLabelRef.current);
        if (classifier.current.getNumClasses() > 0) {
            classifier.current.predictClass(inputTensor).then((result) => {
                if(result.confidences[result.label] > 0.95) {
                    setCurrentGesture(result.label);
                    setConfidence(Math.round(result.confidences[result.label] * 100));
                    handleGestureLogic(result.label);
                }
            });
        }
        inputTensor.dispose();
    } else { setCurrentGesture("No Hand"); setConfidence(0); }
  };

  const handleGestureLogic = (bestClass) => {
      if (bestClass === "SPEAK") {
          if (!actionCooldown.current && sentenceRef.current.length > 0) {
              triggerSpeak(); 
              actionCooldown.current = true; setTimeout(() => { actionCooldown.current = false }, 2000);
          }
          return;
      }
      if (bestClass === "DELETE") {
          if (!actionCooldown.current && sentenceRef.current.length > 0) {
              deleteLastWord(); 
              actionCooldown.current = true; setTimeout(() => { actionCooldown.current = false }, 1500);
          }
          return;
      }
      if(bestClass !== "NOTHING" && !trainingLabelRef.current) {
          if (bestClass === lastWordRef.current) wordStableCount.current += 1;
          else { wordStableCount.current = 0; lastWordRef.current = bestClass; }
          if (wordStableCount.current === 15) { addToSentence(bestClass); wordStableCount.current = 0; }
      }
  };

  const initiateTraining = (labelOverride = null) => {
      const target = labelOverride || customLabel;
      if(!target) return alert("Enter name!");
      let count = prepTime;
      setTimerDisplay(count); setStatus(`Ready... ${count}`);
      const interval = setInterval(() => {
          count--;
          if (count > 0) { setTimerDisplay(count); setStatus(`Ready... ${count}`); } 
          else { clearInterval(interval); setTimerDisplay("GO!"); runTraining(target); }
      }, 1000);
  };

  const runTraining = (label) => {
      setStatus(`Learning '${label}'...`);
      trainingLabelRef.current = label;
      if (!savedGestures.includes(label)) setSavedGestures(prev => [...prev, label]);
      setTimeout(() => {
          trainingLabelRef.current = null; setStatus("Saved.");
          saveBrain(); setTimerDisplay(null); setCustomLabel(""); 
      }, 3000);
  };

  const deleteGesture = (label) => {
      if(!window.confirm(`Delete '${label}'?`)) return;
      try { if (classifier.current.getNumClasses() > 0) classifier.current.clearClass(label); saveBrain(); } catch(e) {}
      setSavedGestures(prev => prev.filter(g => g !== label));
  };

  const triggerSpeak = () => {
      // Safety check for APK
      if (!synth || !utterance) {
          if ('speechSynthesis' in window) {
             synth = window.speechSynthesis;
             utterance = new SpeechSynthesisUtterance();
          } else {
             alert("Speech not supported");
             return;
          }
      }
      
      let text = sentenceRef.current.join(" ");
      if(!text) text = "Make a sentence first."; 

      synth.cancel(); // Reset

      utterance.text = text;
      utterance.volume = 1;
      utterance.rate = 1;
      utterance.pitch = 1;
      
      synth.speak(utterance);
      setStatus("🔊 SPEAKING..."); 
      setTimeout(() => setStatus("Ready"), 2000);
  };

  const deleteLastWord = () => {
      setSentence(prev => {
          if (prev.length === 0) return prev;
          const newArr = [...prev];
          newArr.pop();
          return newArr;
      });
      setStatus("⬅️ Deleted Word");
      setTimeout(() => setStatus("Ready"), 1000);
  };

  const addToSentence = (word) => setSentence(prev => (prev[prev.length - 1] === word ? prev : [...prev, word]));

  if (!isAppStarted) {
      return (
          <div style={styles.startScreen}>
              <h1 style={styles.logoText}>Sign Speak</h1>
              <p style={{color: "#AAA", marginBottom: "30px"}}>AI Assistant Ready</p>
              <button onClick={() => { setIsAppStarted(true); setTimeout(initAI, 1000); }} style={styles.btnBig}>Start Camera 📷</button>
          </div>
      );
  }

  const transformStyle = facingMode === "user" ? "scaleX(-1)" : "none";

  return (
    <div style={styles.mainContainer}>
      <div style={styles.cameraBox}>
            <Webcam 
            key={facingMode}
            ref={webcamRef} 
            videoConstraints={{ 
                facingMode: facingMode,
                width: 640,    // <--- THIS MAKES IT FAST (Resolution)
                height: 480    // <--- THIS MAKES IT FAST
            }} 
            style={{ width: "100%", height: "100%", objectFit: "cover", transform: transformStyle, position: "absolute", top:0, left:0, zIndex: 1 }} 
            audio={false}
        />

        <canvas 
            ref={canvasRef} 
            style={{ width: "100%", height: "100%", objectFit: "cover", transform: transformStyle, position: "absolute", top:0, left:0, zIndex: 2 }} 
        />
        <div style={styles.overlayLabel}>
            <span style={{color: "#00FF00", fontWeight: "bold"}}>{currentGesture}</span> 
            <span style={{color: "#AAA", fontSize: "12px", marginLeft: "5px"}}>{confidence}%</span>
        </div>
        <button onClick={switchCamera} style={styles.switchBtn}>🔄</button>
        {timerDisplay && <div style={styles.timerOverlay}>{timerDisplay}</div>}
      </div>

      <div style={styles.controlsArea}>
          {activeTab === 'train' && (
              <div style={styles.tabContent}>
                  <h2 style={styles.tabTitle}>Training Studio</h2>
                  <div style={styles.inputRow}>
                      <input style={styles.input} placeholder="GESTURE NAME" value={customLabel} onChange={(e) => setCustomLabel(e.target.value.toUpperCase())} />
                      <button style={styles.btnAction} onClick={() => initiateTraining()}>TEACH</button>
                  </div>
                  <div style={styles.grid2}>
                      <button style={styles.btnNeutral} onClick={() => initiateTraining("NOTHING")}>✋ Teach "NOTHING"</button>
                      <button style={styles.btnDelete} onClick={() => initiateTraining("DELETE")}>⬅️ Teach "DELETE"</button>
                  </div>
              </div>
          )}
          {activeTab === 'speak' && (
              <div style={styles.tabContent}>
                  <h2 style={{color: "#00FF00", marginBottom: "10px"}}>Speech Mode</h2>
                  <div style={styles.textBox}>{sentence.length === 0 ? "..." : sentence.join(" ")}</div>
                  <div style={styles.grid3}>
                      <button onClick={triggerSpeak} style={styles.btnSpeak}>🔊 SPEAK</button>
                      <button onClick={deleteLastWord} style={styles.btnDelWord}>⬅️ DEL</button>
                      <button onClick={() => setSentence([])} style={styles.btnClear}>🗑️ ALL</button>
                  </div>
                  <p style={{color:"#666", marginTop: "10px", fontSize:"12px"}}>{status}</p>
              </div>
          )}
          {activeTab === 'library' && (
              <div style={styles.tabContent}>
                  <h2 style={{color: "orange", marginBottom: "10px"}}>Library</h2>
                  <div style={styles.chipContainer}>
                      {savedGestures.map(g => (<div key={g} style={styles.chip}><span>{g}</span><button onClick={() => deleteGesture(g)} style={styles.deleteX}>✕</button></div>))}
                  </div>
                  <hr style={{width: "100%", borderColor: "#333", margin: "15px 0"}}/>
                  <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", width: "100%"}}>
                      {ASL_SUGGESTIONS.map(word => (<button key={word} onClick={() => { setCustomLabel(word); setActiveTab('train'); initiateTraining(word); }} style={styles.aslBtn}>{word}</button>))}
                  </div>
              </div>
          )}
      </div>
      
      <div style={styles.navBar}>
          <button onClick={() => setActiveTab('train')} style={{...styles.navBtn, color: activeTab === 'train' ? "#0088FF" : "#666"}}>🛠️ Train</button>
          <button onClick={() => setActiveTab('speak')} style={{...styles.navBtn, color: activeTab === 'speak' ? "#00FF00" : "#666"}}>💬 Speak</button>
          <button onClick={() => setActiveTab('library')} style={{...styles.navBtn, color: activeTab === 'library' ? "orange" : "#666"}}>📚 Lib</button>
      </div>
    </div>
  );
}

// STYLES
const styles = {
    mainContainer: { position: "absolute", top:0, left:0, width:"100%", height:"100%", background:"#121212", display:"flex", flexDirection:"column", fontFamily:"'Segoe UI', sans-serif" },
    cameraBox: { flex: "1", position: "relative", background: "#000", overflow: "hidden", maxHeight:"45%" },
    switchBtn: { position: "absolute", top: 15, right: 15, background: "rgba(255,255,255,0.2)", backdropFilter: "blur(5px)", border: "1px solid rgba(255,255,255,0.3)", borderRadius: "50%", width: "45px", height: "45px", fontSize: "20px", color: "white", zIndex: 50 },
    overlayLabel: { position: "absolute", bottom: 15, left: 15, background: "rgba(0,0,0,0.7)", padding: "8px 15px", borderRadius: "20px", backdropFilter: "blur(4px)", zIndex: 40, border: "1px solid #333" },
    timerOverlay: { position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", fontSize: "100px", fontWeight: "bold", color: "#00FFFF", textShadow: "0 0 20px rgba(0,255,255,0.5)", zIndex: 60 },
    controlsArea: { flex: "1", padding: "20px", overflowY: "auto", paddingBottom: "90px" },
    tabContent: { display: "flex", flexDirection: "column", alignItems: "center", width: "100%" },
    tabTitle: { color: "#0088FF", marginBottom: "15px" },
    inputRow: { display: "flex", width: "100%", gap: "10px", marginBottom: "15px" },
    grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", width: "100%" },
    grid3: { display: "grid", gridTemplateColumns: "2fr 1fr 1fr", gap: "10px", width: "100%" },
    btnAction: { background: "linear-gradient(135deg, #0088FF, #0055FF)", border: "none", color: "white", padding: "15px", borderRadius: "12px", fontWeight: "bold", fontSize: "14px", boxShadow: "0 4px 15px rgba(0,136,255,0.3)" },
    input: { flex: 1, background: "#222", border: "1px solid #444", borderRadius: "12px", padding: "15px", color: "white", outline: "none" },
    btnNeutral: { background: "#333", color: "#AAA", border: "1px solid #444", padding: "15px", borderRadius: "12px", fontWeight: "bold" },
    btnDelete: { background: "#3D2B2B", color: "#FF8888", border: "1px solid #553333", padding: "15px", borderRadius: "12px", fontWeight: "bold" },
    
    // BIGGER BUTTONS
    btnSpeak: { background: "linear-gradient(135deg, #00CC00, #008800)", color: "white", border: "none", borderRadius: "12px", fontWeight: "bold", boxShadow: "0 4px 15px rgba(0,200,0,0.3)", padding: "20px", fontSize: "18px" },
    btnDelWord: { background: "#CC7700", color: "white", border: "none", borderRadius: "12px", fontWeight: "bold", padding: "20px", fontSize: "18px" },
    btnClear: { background: "#CC0000", color: "white", border: "none", borderRadius: "12px", fontWeight: "bold", padding: "20px", fontSize: "18px" },
    
    btnPrimary: { background: "#0088FF", color: "white", border: "none", padding: "15px", borderRadius: "10px", fontWeight: "bold" },
    textBox: { width: "100%", minHeight: "80px", background: "#1A1A1A", border: "1px solid #333", borderRadius: "15px", padding: "20px", fontSize: "22px", color: "white", marginBottom: "20px", display: "flex", alignItems: "center", justifyContent: "center" },
    chipContainer: { display: "flex", flexWrap: "wrap", gap: "10px", justifyContent: "center" },
    chip: { background: "#252525", padding: "8px 16px", borderRadius: "20px", border: "1px solid #444", fontSize: "14px", color: "#DDD", display: "flex", gap: "10px" },
    deleteX: { background: "none", border: "none", color: "#FF4444", fontWeight: "bold", cursor: "pointer" },
    aslBtn: { background: "#222", border: "1px solid #333", color: "#AAA", padding: "12px", borderRadius: "8px" },
    navBar: { position: "fixed", bottom: 0, left: 0, width: "100%", height: "70px", background: "#161616", borderTop: "1px solid #333", display: "flex", justifyContent: "space-around", alignItems: "center", zIndex: 100 },
    navBtn: { background: "none", border: "none", fontSize: "16px", fontWeight: "600", padding: "10px" },
    startScreen: { position: "absolute", top:0, left:0, width:"100%", height:"100%", background: "#000", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" },
    logoText: { fontSize: "48px", fontWeight: "bold", background: "-webkit-linear-gradient(#00FFFF, #0088FF)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: "10px" },
    btnBig: { padding: "20px 40px", fontSize: "20px", borderRadius: "50px", border: "none", background: "white", color: "black", fontWeight: "bold", boxShadow: "0 0 20px rgba(255,255,255,0.2)" }
};














