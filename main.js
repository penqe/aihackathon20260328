import * as THREE from 'three';
import { HandLandmarker, ImageClassifier, FilesetResolver } from '@mediapipe/tasks-vision';

const statusEl = document.getElementById('status');
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('canvas3d');

// --- 1. Three.js Setup ---
const scene = new THREE.Scene();

// We want the canvas to overlay the video perfectly
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 4;

const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);

const geometry = new THREE.BoxGeometry(2, 2, 2); // 削除してしまったジオメトリを復元
const material = new THREE.MeshPhysicalMaterial({
  color: 0xffffff,
  metalness: 0.1, 
  roughness: 0.0, 
  transmission: 1.0, 
  ior: 1.5, 
  thickness: 2.5, 
  transparent: true,
  opacity: 1.0,
  clearcoat: 1.0, // 表面の光沢反射（シェーディング強調）
  clearcoatRoughness: 0.1,
  wireframe: false,
});
const cube = new THREE.Mesh(geometry, material);

// 立方体であることを明確にするため、半透明の白い「輪郭線（エッジ）」を追加
const edgesGeometry = new THREE.EdgesGeometry(geometry);
const edgesMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 });
const cubeEdges = new THREE.LineSegments(edgesGeometry, edgesMaterial);
cube.add(cubeEdges);

scene.add(cube);

// --- 爆発エフェクト用の破片キューブ群（3x3x3 = 27個） ---
let isExploded = false;
let explosionTimer = 0;
const explosionPieces = new THREE.Group();
const pieceVelocities = [];

// 破片のサイズは全体の1/3
const pieceGeo = new THREE.BoxGeometry(2/3, 2/3, 2/3);
for (let x = -1; x <= 1; x++) {
  for (let y = -1; y <= 1; y++) {
    for (let z = -1; z <= 1; z++) {
      const p = new THREE.Mesh(pieceGeo, material);
      // グリッド状の初期位置に並べる
      p.position.set(x * (2/3), y * (2/3), z * (2/3));
      p.userData.originalPos = p.position.clone();
      p.userData.spin = new THREE.Vector3();
      
      const pEdges = new THREE.LineSegments(new THREE.EdgesGeometry(pieceGeo), edgesMaterial);
      p.add(pEdges);
      
      explosionPieces.add(p);
      pieceVelocities.push(new THREE.Vector3());
    }
  }
}
// --------------------------------------------------------

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 1);
dirLight.position.set(5, 5, 5);
scene.add(dirLight);

// Handle resize & keep video background cropped perfectly
let videoTexture = null;

function alignVideoBackground() {
  if (!videoTexture || !videoEl.videoWidth) return;
  const videoAspect = videoEl.videoWidth / videoEl.videoHeight;
  const windowAspect = window.innerWidth / window.innerHeight;
  
  if (windowAspect > videoAspect) {
    // 画面の方が横長：ビデオの上下をカット
    const scale = videoAspect / windowAspect;
    videoTexture.repeat.set(-1, scale);
    videoTexture.offset.set(1, (1 - scale) / 2);
  } else {
    // 画面の方が縦長：ビデオの左右をカット
    const scale = windowAspect / videoAspect;
    videoTexture.repeat.set(-scale, 1);
    videoTexture.offset.set((1 + scale) / 2, 0);
  }
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  alignVideoBackground();
});

// --- 2. MediaPipe Setup & Camera ---
let handLandmarker = null;
let imageClassifier = null; // 画像分類モデル
let webcamRunning = false;
let lastVideoTime = -1;

// ImageClassifier用のオフスクリーンキャンバス（キューブの裏側だけを切り抜いて判定するため）
const cropCanvas = document.createElement('canvas');
cropCanvas.width = 224;
cropCanvas.height = 224;
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
let lastClassifyTime = 0;

async function initMediaPipe() {
  statusEl.innerText = "MediaPipe モデルを読み込み中...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  // 物体認識用モデルのロード
  imageClassifier = await ImageClassifier.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite`
    },
    maxResults: 1,
    runningMode: "IMAGE" // Canvasの静止画像を都度渡すのでIMAGEモード
  });

  statusEl.innerText = "Webカメラを待機中... (ブラウザの許可が必要です)";
  startCamera();
}

function startCamera() {
  navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" } })
    .then((stream) => {
      videoEl.srcObject = stream;
      videoEl.addEventListener("loadeddata", () => {
        webcamRunning = true;
        
        // 1. 真の屈折（Refraction）のために、ビデオをThree.jsのシーン背景に設定
        videoTexture = new THREE.VideoTexture(videoEl);
        videoTexture.colorSpace = THREE.SRGBColorSpace;
        scene.background = videoTexture;
        alignVideoBackground(); // CSSのobject-fit:coverと同じようにクロップして合わせる
        
        // 2. 表面への映り込み（Reflection）のための環境マップも別途作成
        const envTexture = new THREE.VideoTexture(videoEl);
        envTexture.colorSpace = THREE.SRGBColorSpace;
        envTexture.mapping = THREE.EquirectangularReflectionMapping; 
        envTexture.wrapS = THREE.RepeatWrapping;
        envTexture.repeat.x = -1; // 反転
        
        cube.material.envMap = envTexture;
        cube.material.envMapIntensity = 1.0; 
        cube.material.needsUpdate = true;
        
        statusEl.innerText = "準備完了！手をカメラに映して操作してください。";
        renderLoop();
      });
    })
    .catch((err) => {
      console.error(err);
      statusEl.innerText = "カメラの起動に失敗しました。カメラのアクセス権限を確認してください。";
    });
}

// --- 3. Interaction Logic ---
let isOneHandSwiping = false; // 人差し指スワイプ（回転）
let isOneHandPinching = false; // 片手ピンチ（3D移動）
let isTwoHandsPinching = false; // 両手ピンチ（拡大縮小＋回転）
let previousIndexTip = { x: 0, y: 0 };
let previousPinchDistance = 0;
let previousPinchCenter = { x: 0, y: 0, zSize: 0 };
let velocity = { x: 0, y: 0 }; // 回転の慣性
let positionVelocity = { x: 0, y: 0, z: 0 }; // 移動の慣性
const PINCH_THRESHOLD = 0.05; // Distance representing a pinch

function processHands(results) {
  const landmarksArray = results.landmarks;
  
  if (landmarksArray && landmarksArray.length > 0) {
    // --- 【追加】手が「パー（大きく開いている）」かどうかを判定して爆発させる ---
    let isOpenHand = false;
    for (let i = 0; i < landmarksArray.length; i++) {
      const hand = landmarksArray[i];
      // 手のひらのサイズ（手首〜中指の付け根）
      const palmSize = Math.sqrt(Math.pow(hand[0].x - hand[9].x, 2) + Math.pow(hand[0].y - hand[9].y, 2));
      // 指の伸び具合（手首から各指先までの距離の合計）
      let fingersLength = 0;
      for (const tip of [8, 12, 16, 20]) {
        fingersLength += Math.sqrt(Math.pow(hand[0].x - hand[tip].x, 2) + Math.pow(hand[0].y - hand[tip].y, 2));
      }
      // 手のひらに対する指の割合でパーを判定
      if ((fingersLength / palmSize) > 7.5) {
        isOpenHand = true;
        break;
      }
    }
    
    if (isOpenHand && !isExploded) {
      isExploded = true;
      explosionTimer = 60; // 約1秒間は飛散状態を強制
      
      cube.visible = false;
      scene.add(explosionPieces);
      
      explosionPieces.children.forEach((p, index) => {
         p.position.copy(p.userData.originalPos);
         // 中心から外に向かうベクトルを計算し、ランダムな初速を与える
         const vel = p.userData.originalPos.clone().normalize().multiplyScalar(0.2 + Math.random() * 0.3);
         pieceVelocities[index] = vel;
         // ランダムな自転トルク
         p.userData.spin.set((Math.random()-0.5)*0.5, (Math.random()-0.5)*0.5, (Math.random()-0.5)*0.5);
         p.rotation.set(0, 0, 0);
      });
    }
    // -------------------------------------------------------------

    // 1. 各手のピンチ状態をチェック
    const pinchingHands = [];
    
    for (let i = 0; i < landmarksArray.length; i++) {
      const hand = landmarksArray[i];
      const indexTip = hand[8];
      const thumbTip = hand[4];
      
      const dx = indexTip.x - thumbTip.x;
      const dy = indexTip.y - thumbTip.y;
      const dz = indexTip.z - thumbTip.z;
      const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
      
      if (distance < PINCH_THRESHOLD) {
        // 奥行きの基準として、手首(0)と中指付け根(9)の距離を計算（遠ざかると小さくなる）
        const wrist = hand[0];
        const middleMCP = hand[9];
        const handSize = Math.sqrt(Math.pow(wrist.x - middleMCP.x, 2) + Math.pow(wrist.y - middleMCP.y, 2));
        
        pinchingHands.push({
          x: (indexTip.x + thumbTip.x) / 2,
          y: (indexTip.y + thumbTip.y) / 2,
          zSize: handSize
        });
      }
    }
    
    // 2. ピンチしている手の数に応じて処理を変える
    if (pinchingHands.length === 2) {
      if (isOneHandSwiping) isOneHandSwiping = false;
      
      const p1 = pinchingHands[0];
      const p2 = pinchingHands[1];
      const dx = p1.x - p2.x;
      const dy = p1.y - p2.y;
      const currentDistance = Math.sqrt(dx*dx + dy*dy);
      
      // 両手の中間座標を計算（回転に使う）
      const currentMidpoint = {
        x: (p1.x + p2.x) / 2,
        y: (p1.y + p2.y) / 2
      };
      
      if (!isTwoHandsPinching) {
        isTwoHandsPinching = true;
        previousPinchDistance = currentDistance;
        previousPinchCenter = currentMidpoint;
        velocity = { x: 0, y: 0 };
        // 発光による色の変化を削除
      } else {
        // 【スケール処理】
        const deltaDistance = currentDistance - previousPinchDistance;
        const scaleChange = deltaDistance * 5;
        let newScale = cube.scale.x + scaleChange;
        newScale = Math.max(0.2, Math.min(newScale, 10.0));
        cube.scale.set(newScale, newScale, newScale);
        
        // 【回転処理】
        const deltaX = currentMidpoint.x - previousPinchCenter.x;
        const deltaY = currentMidpoint.y - previousPinchCenter.y;
        
        velocity.y = deltaX * 8;
        velocity.x = -deltaY * 8;
        
        cube.rotation.y += velocity.y;
        cube.rotation.x += velocity.x;
        
        previousPinchDistance = currentDistance;
        previousPinchCenter = currentMidpoint;
      }
    } else if (pinchingHands.length === 1) {
      // 【片手ピンチ：3D移動】
      if (isOneHandSwiping) isOneHandSwiping = false;
      if (isTwoHandsPinching) isTwoHandsPinching = false;
      
      const pinchCenter = pinchingHands[0];
      const currentPos = { x: pinchCenter.x, y: pinchCenter.y, zSize: pinchCenter.zSize };
      
      // フレーム飛散防止（手が跳んだ場合は無視）
      let isJump = false;
      if (isOneHandPinching) {
        const jumpDist = Math.sqrt(Math.pow(currentPos.x - previousPinchCenter.x, 2) + Math.pow(currentPos.y - previousPinchCenter.y, 2));
        if (jumpDist > 0.15) isJump = true;
      }
      
      if (!isOneHandPinching || isJump) {
        isOneHandPinching = true;
        previousPinchCenter = currentPos;
        if (!isJump) positionVelocity = { x: 0, y: 0, z: 0 }; // 移動の慣性をリセット
        // 発光による色の変化を削除
      } else {
        const deltaX = currentPos.x - previousPinchCenter.x;
        const deltaY = currentPos.y - previousPinchCenter.y;
        const deltaZ = currentPos.zSize - previousPinchCenter.zSize;
        
        positionVelocity.x = -deltaX * 15; // 左右反転
        positionVelocity.y = -deltaY * 15; // 上下反転
        positionVelocity.z = deltaZ * 30; // サイズが大きくなる（近づく）＝Zプラス
        
        // 異常な速度はカットして飛散を防止
        positionVelocity.x = Math.max(-0.5, Math.min(0.5, positionVelocity.x));
        positionVelocity.y = Math.max(-0.5, Math.min(0.5, positionVelocity.y));
        positionVelocity.z = Math.max(-0.5, Math.min(0.5, positionVelocity.z));
        
        cube.position.x += positionVelocity.x;
        cube.position.y += positionVelocity.y;
        cube.position.z += positionVelocity.z;
        
        previousPinchCenter = currentPos;
      }
    } else if (landmarksArray.length === 1) {
      // 【ピンチしていない片手：人差し指スワイプで回転】
      if (isTwoHandsPinching) isTwoHandsPinching = false;
      if (isOneHandPinching) isOneHandPinching = false;
      
      const indexTip = landmarksArray[0][8]; 
      const currentPos = { x: indexTip.x, y: indexTip.y };
      
      // ジャンプ判定
      let isJump = false;
      if (isOneHandSwiping) {
        const jumpDist = Math.sqrt(Math.pow(currentPos.x - previousIndexTip.x, 2) + Math.pow(currentPos.y - previousIndexTip.y, 2));
        if (jumpDist > 0.15) isJump = true;
      }
      
      if (!isOneHandSwiping || isJump) {
        isOneHandSwiping = true;
        previousIndexTip = currentPos;
        if (!isJump) velocity = { x: 0, y: 0 };
        // 発光による色の変化を削除
      } else {
        const deltaX = currentPos.x - previousIndexTip.x;
        const deltaY = currentPos.y - previousIndexTip.y;
        
        velocity.y = deltaX * 12;
        velocity.x = -deltaY * 12;
        
        cube.rotation.y += velocity.y;
        cube.rotation.x += velocity.x;
        
        previousIndexTip = currentPos;
      }
    } else {
      resetInteractionState();
    }
  } else {
    resetInteractionState();
  }
}

function resetInteractionState() {
  if (isOneHandSwiping || isOneHandPinching || isTwoHandsPinching) {
    isOneHandSwiping = false;
    isOneHandPinching = false;
    isTwoHandsPinching = false;
    // 発光による色の変化を削除
  }
}

// --- 4. Render Loop ---
let trembleTimer = 0; // 震えアニメーション用タイマー
let isRecovering = false; // 消失時復帰フラグ
const frustum = new THREE.Frustum();
const projScreenMatrix = new THREE.Matrix4();

function renderLoop() {
  requestAnimationFrame(renderLoop);

  // 【保護機構】NaNなどによる致命的エラーでの永遠の消失を防ぐ
  if (isNaN(cube.position.x) || isNaN(cube.position.y) || isNaN(cube.position.z)) {
    cube.position.set(0, 0, 0);
    positionVelocity = { x: 0, y: 0, z: 0 };
    velocity = { x: 0, y: 0 };
  }

  // 【消失検知】カメラの視野（フラストム）とキューブの交差判定
  camera.updateMatrixWorld();
  projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  frustum.setFromProjectionMatrix(projScreenMatrix);
  
  if (!frustum.intersectsObject(cube)) {
    // キューブがカメラの視界から完全に消え去った
    isRecovering = true;
  }

  // Detect Hands if there is new video frame
  if (handLandmarker && webcamRunning && videoEl.currentTime !== lastVideoTime) {
    lastVideoTime = videoEl.currentTime;
    const results = handLandmarker.detectForVideo(videoEl, performance.now());
    // 復帰中はユーザーの操作を受け付けない
    if (!isRecovering) {
      processHands(results);
    }
  }

  // 【物体認識】0.5秒おきにキューブの裏側に何があるかをリアルタイム判定する
  if (imageClassifier && webcamRunning && performance.now() - lastClassifyTime > 500 && !isRecovering) {
    lastClassifyTime = performance.now();
    
    // キューブの3D座標からスクリーン座標[-1, 1]を取得
    const tempV = cube.position.clone();
    tempV.project(camera);
    
    // ビデオと画面のアスペクト比のズレを考慮して、正確なピクセル座標へ変換
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    if (vw > 0 && vh > 0) {
      const windowAspect = window.innerWidth / window.innerHeight;
      const videoAspect = vw / vh;
      let videoX, videoY, videoScale;
      
      if (windowAspect > videoAspect) {
        // 画面が横長の場合（上下が見切れている）
        videoX = (0.5 - tempV.x / 2) * vw; // カメラが左右反転しているためXも反転
        const visibleHeight = vw / windowAspect;
        const screenYNorm = (1.0 - tempV.y) / 2;
        videoY = (vh - visibleHeight)/2 + screenYNorm * visibleHeight;
        videoScale = vw / window.innerWidth;
      } else {
        // 画面が縦長の場合（左右が見切れている）
        const visibleWidth = vh * windowAspect;
        const screenXNorm = 0.5 - tempV.x / 2;
        videoX = (vw - visibleWidth)/2 + screenXNorm * visibleWidth;
        const screenYNorm = (1.0 - tempV.y) / 2;
        videoY = screenYNorm * vh;
        videoScale = vh / window.innerHeight;
      }
      
      // キューブの3Dスケール（見かけの大きさ）に応じてクロップ範囲を変える
      const cropSize = Math.max(100, 150 * cube.scale.x) * videoScale;
      
      cropCtx.clearRect(0, 0, 224, 224);
      cropCtx.drawImage(
        videoEl, 
        videoX - cropSize/2, videoY - cropSize/2, cropSize, cropSize, 
        0, 0, 224, 224
      );
      
      // クロップした画像をEfficientNetで判定
      const predictionResult = imageClassifier.classify(cropCanvas);
      if (predictionResult && predictionResult.classifications.length > 0 && predictionResult.classifications[0].categories.length > 0) {
        const category = predictionResult.classifications[0].categories[0];
        const predictionEl = document.getElementById('prediction');
        if (predictionEl) {
          // EfficientNetの予測名とパーセンテージを表示
          predictionEl.innerText = `🔍 見出しているもの: ${category.categoryName} (${Math.round(category.score * 100)}%)`;
        }
      }
    }
  }

  if (isRecovering) {
    // 復帰モード：画面中央へ吸い込まれるようにスムーズに戻ってくる
    cube.position.lerp(new THREE.Vector3(0, 0, 0), 0.1);
    positionVelocity = { x: 0, y: 0, z: 0 }; // 変な移動慣性を打ち消す
    
    // 復帰中はクルクル回りながら戻ってくる演出
    cube.rotation.y += 0.05;
    cube.rotation.x += 0.03;
    
    // 中央付近に戻ったら復帰完了、操作受け付け再開
    if (cube.position.length() < 0.5) {
      isRecovering = false;
    }
  } else {
    // 回転の慣性
    if (!isOneHandSwiping && !isTwoHandsPinching) {
      cube.rotation.y += velocity.y;
      cube.rotation.x += velocity.x;
      velocity.x *= 0.95;
      velocity.y *= 0.95;
      if (Math.abs(velocity.x) < 0.0001) velocity.x = 0;
      if (Math.abs(velocity.y) < 0.0001) velocity.y = 0;
      
      // 【生物感の表現】デフォルトでゆっくりとY軸で自転する
      cube.rotation.y += 0.005;
    }
    
    // 移動の慣性と緩いバウンダリ
    if (!isOneHandPinching) {
      cube.position.x += positionVelocity.x;
      cube.position.y += positionVelocity.y;
      cube.position.z += positionVelocity.z;
      positionVelocity.x *= 0.90; // 移動は少し早めに止まるように強めの摩擦
      positionVelocity.y *= 0.90;
      positionVelocity.z *= 0.90;
      
      // 【紛失防止】画面外やカメラの裏に行き過ぎた場合、中央へ緩やかに引き戻す
      if (cube.position.z > 2) cube.position.z += (2 - cube.position.z) * 0.1;
      if (cube.position.z < -8) cube.position.z += (-8 - cube.position.z) * 0.1;
      if (cube.position.x > 5) cube.position.x += (5 - cube.position.x) * 0.1;
      if (cube.position.x < -5) cube.position.x += (-5 - cube.position.x) * 0.1;
      if (cube.position.y > 4) cube.position.y += (4 - cube.position.y) * 0.1;
      if (cube.position.y < -4) cube.position.y += (-4 - cube.position.y) * 0.1;
    }
  }

  // 【生物感の表現】時々、意思を持つように身震いする
  // ※ユーザーが操作していない（手が触れていない）間だけ発生する
  if (!isOneHandPinching && !isOneHandSwiping && !isTwoHandsPinching) {
    if (Math.random() < 0.005) { // 0.5% の確率で発動（数秒に1回程度の感覚）
      trembleTimer = Math.floor(10 + Math.random() * 20); // 10〜30フレーム継続
    }
  }

  let tx = 0, ty = 0, tz = 0;
  if (trembleTimer > 0) {
    // 激しく微小に移動させて震えを表現
    tx = (Math.random() - 0.5) * 0.15;
    ty = (Math.random() - 0.5) * 0.15;
    tz = (Math.random() - 0.5) * 0.15;
    cube.position.x += tx;
    cube.position.y += ty;
    cube.position.z += tz;
    trembleTimer--;
  }

  // 【爆発エフェクトの更新と復元】
  if (isExploded) {
    // 破片のグループ全体の位置・回転・スケールを、見えないメインオブジェクトと同期する
    // これにより、破片になった状態でもユーザーはつまんで移動させたり回したりできる！
    explosionPieces.position.copy(cube.position);
    explosionPieces.rotation.copy(cube.rotation);
    explosionPieces.scale.copy(cube.scale);
    
    let allSettled = true;
    if (explosionTimer > 0) explosionTimer--;
    
    explosionPieces.children.forEach((p, i) => {
      // 破片自身のローカルな移動と回転を適用
      p.position.add(pieceVelocities[i]);
      p.rotation.x += p.userData.spin.x;
      p.rotation.y += p.userData.spin.y;
      p.rotation.z += p.userData.spin.z;
      
      // 空気抵抗で減速
      pieceVelocities[i].multiplyScalar(0.92);
      p.userData.spin.multiplyScalar(0.92);
      
      // タイマーが切れたら元の位置（27個のグリッド）に磁石のように吸い寄せられる
      if (explosionTimer <= 0) {
        p.position.lerp(p.userData.originalPos, 0.08); // 中心にスッと戻る
        p.rotation.x *= 0.9;
        p.rotation.y *= 0.9;
        p.rotation.z *= 0.9;
        
        // 全ての破片が完全に元の位置に戻ったかチェック
        if (p.position.distanceTo(p.userData.originalPos) > 0.01 || Math.abs(p.rotation.x) > 0.05) {
          allSettled = false;
        }
      } else {
        allSettled = false;
      }
    });
    
    // 全てが元に戻り切ったら、元の1個のキューブに差し替える
    if (allSettled && explosionTimer <= 0) {
      isExploded = false;
      scene.remove(explosionPieces);
      cube.visible = true; // メインフレームを復活
    }
  }

  renderer.render(scene, camera);
  
  // 描画後、座標を正確に戻す（元の位置から少しずつズレて「ドリフト」していくのを完全に防ぐ）
  if (tx !== 0) {
    cube.position.x -= tx;
    cube.position.y -= ty;
    cube.position.z -= tz;
  }
}

// Start
initMediaPipe();
