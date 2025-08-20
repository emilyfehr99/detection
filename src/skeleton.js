// Lightweight client-side skeleton overlay helpers
function updatePlayerSelect(frameNum) {
  const frameData = framesData[frameNum];
  const select = document.getElementById('playerSelect');
  if (!frameData || !select) return;
  // Prefer players visible in the current frame (type == 'player')
  let ids = (frameData.players || [])
    .filter(p => p.type && p.type.toLowerCase() === 'player')
    .map(p => p.player_id);

  // Fallback: use global list of player IDs seen anywhere
  if (!ids || ids.length === 0) {
    if (!window.skeletonGlobalPlayerIds) {
      const set = new Set();
      (framesData || []).forEach(f => {
        (f.players || []).forEach(p => {
          if (p.type && p.type.toLowerCase() === 'player') set.add(p.player_id);
        });
      });
      window.skeletonGlobalPlayerIds = Array.from(set);
    }
    ids = window.skeletonGlobalPlayerIds;
  }
  const prev = select.value;
  select.innerHTML = ids.map(id => `<option value="${id}">${id}</option>`).join('');
  if (ids.includes(prev)) {
    select.value = prev;
  } else if (ids.length > 0) {
    select.value = ids[0];
  }
}

function drawSkeletonForSelectedPlayer(frameNum) {
  const frameData = framesData[frameNum];
  const select = document.getElementById('playerSelect');
  const img = document.getElementById('skeletonFrame');
  const canvas = document.getElementById('skeletonCanvas');
  if (!frameData || !select || !img || !canvas) return;
  const playerId = select.value;
  const player = (frameData.players || []).find(p => p.player_id === playerId);
  const landmarks = player && player.pose_landmarks;

  const rect = img.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks || !Array.isArray(landmarks)) return;

  const natW = img.naturalWidth || rect.width;
  const natH = img.naturalHeight || rect.height;
  const scaleX = rect.width / natW;
  const scaleY = rect.height / natH;

  const C = { NOSE:0, LEFT_SHOULDER:11, RIGHT_SHOULDER:12, LEFT_ELBOW:13, RIGHT_ELBOW:14, LEFT_WRIST:15, RIGHT_WRIST:16, LEFT_HIP:23, RIGHT_HIP:24, LEFT_KNEE:25, RIGHT_KNEE:26, LEFT_ANKLE:27, RIGHT_ANKLE:28, LEFT_HEEL:29, RIGHT_HEEL:30, LEFT_FOOT_INDEX:31, RIGHT_FOOT_INDEX:32 };
  const pairs = [
    [C.LEFT_SHOULDER, C.RIGHT_SHOULDER],
    [C.LEFT_SHOULDER, C.LEFT_ELBOW],
    [C.LEFT_ELBOW, C.LEFT_WRIST],
    [C.RIGHT_SHOULDER, C.RIGHT_ELBOW],
    [C.RIGHT_ELBOW, C.RIGHT_WRIST],
    [C.LEFT_SHOULDER, C.LEFT_HIP],
    [C.RIGHT_SHOULDER, C.RIGHT_HIP],
    [C.LEFT_HIP, C.RIGHT_HIP],
    [C.LEFT_HIP, C.LEFT_KNEE],
    [C.LEFT_KNEE, C.LEFT_ANKLE],
    [C.RIGHT_HIP, C.RIGHT_KNEE],
    [C.RIGHT_KNEE, C.RIGHT_ANKLE]
  ];

  const pts = {};
  landmarks.forEach(lm => { pts[lm.index] = lm; });

  ctx.lineWidth = 3;
  ctx.strokeStyle = '#00ff88';

  // Per-joint colors (RGB)
  const jointColors = {
    0:  [255, 255, 255],   // nose (white)
    11: [255, 255, 255],   // left shoulder (white)
    12: [255, 255, 255],   // right shoulder (white)
    13: [255, 255, 255],   // left elbow (white)
    14: [255, 255, 255],   // right elbow (white)
    15: [255, 255, 255],   // left wrist (white)
    16: [255, 255, 255],   // right wrist (white)
    23: [153, 50, 204],    // left hip (purple)
    24: [153, 50, 204],    // right hip (purple)
    25: [255, 105, 180],   // left knee (pink)
    26: [255, 105, 180],   // right knee (pink)
    27: [0, 255, 0],       // left ankle (green)
    28: [0, 255, 0],       // right ankle (green)
    29: [0, 0, 255],       // left heel (blue)
    30: [0, 0, 255],       // right heel (blue)
    31: [255, 255, 255],   // left foot index (white)
    32: [255, 255, 255]    // right foot index (white)
  };

  // Draw joints
  for (const k in pts) {
    const p = pts[k];
    if (p.visibility !== undefined && p.visibility < 0.3) continue;
    const jc = jointColors.hasOwnProperty(k*1) ? jointColors[k*1] : [0, 255, 136];
    ctx.fillStyle = `rgb(${jc[0]}, ${jc[1]}, ${jc[2]})`;
    ctx.beginPath();
    ctx.arc(p.x * scaleX, p.y * scaleY, 3, 0, 2*Math.PI);
    ctx.fill();
  }
  // Draw bones
  pairs.forEach(([a,b]) => {
    const p1 = pts[a];
    const p2 = pts[b];
    if (!p1 || !p2) return;
    if ((p1.visibility !== undefined && p1.visibility < 0.3) || (p2.visibility !== undefined && p2.visibility < 0.3)) return;
    ctx.beginPath();
    ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
    ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
    ctx.stroke();
  });
}

// Hook player select changes if present
window.addEventListener('DOMContentLoaded', () => {
  const playerSelect = document.getElementById('playerSelect');
  if (playerSelect) {
    playerSelect.addEventListener('change', () => {
      const frameSlider = document.getElementById('frameSlider');
      const frameNum = parseInt(frameSlider.value);
      drawSkeletonForSelectedPlayer(frameNum);
    });
  }
  const goBtn = document.getElementById('skeletonGo');
  if (goBtn) {
    goBtn.addEventListener('click', () => {
      const frameSlider = document.getElementById('frameSlider');
      const frameNum = parseInt(frameSlider.value);
      drawSkeletonForSelectedPlayer(frameNum);
    });
  }
});


