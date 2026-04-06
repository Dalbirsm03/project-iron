// Global Scene Setup
const container = document.getElementById('canvas-3d');
const video = document.getElementById('video-player');
const frameDisplay = document.getElementById('frame-display');

// Failsafe check
if (!container || !video) {
    console.error("CRITICAL ERROR: DOM elements not found. Check your HTML structure.");
}

const scene = new THREE.Scene();
// Add a helper grid so you can see the 3D space even if data fails
const gridHelper = new THREE.GridHelper(500, 50);
scene.add(gridHelper);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 2 / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth / 2, window.innerHeight);
container.appendChild(renderer.domElement);

// Center the camera based on the mock pipeline data boundaries 
// (X ranges 100-500, Y ranges 150-400)
camera.position.set(300, 275, 800); // Pull Z way back to widen the Field of View
camera.lookAt(300, 275, 0);         // Point exactly at the center of the data mass

const trajectoryLines = [];
const pointClouds = [];

function buildTrajectories(data) {
    data.tracks.forEach(track => {
        const points = [];
        track.path.forEach(pt => {
            points.push(new THREE.Vector3(pt.x, pt.y, pt.z));
        });

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 });
        const line = new THREE.Line(geometry, material);
        
        line.geometry.setDrawRange(0, 0); 
        scene.add(line);
        trajectoryLines.push({ line: line, data: track.path });

        const dotGeo = new THREE.BufferGeometry();
        dotGeo.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0], 3));
        const dotMat = new THREE.PointsMaterial({ color: 0xff0000, size: 8 });
        const dot = new THREE.Points(dotGeo, dotMat);
        scene.add(dot);
        pointClouds.push({ dot: dot, data: track.path });
    });
}

function sync3DWithVideo() {
    const currentFrame = Math.floor(video.currentTime * mockData.fps);
    frameDisplay.innerText = currentFrame;

    for (let i = 0; i < trajectoryLines.length; i++) {
        const lineObj = trajectoryLines[i];
        const dotObj = pointClouds[i];
        const path = lineObj.data;

        const targetIdx = Math.min(currentFrame, path.length - 1);

        if (targetIdx >= 0) {
            lineObj.line.geometry.setDrawRange(0, targetIdx);

            const currentPos = path[targetIdx];
            const positions = dotObj.dot.geometry.attributes.position.array;
            positions[0] = currentPos.x;
            positions[1] = currentPos.y;
            positions[2] = currentPos.z;
            dotObj.dot.geometry.attributes.position.needsUpdate = true;
        }
    }
}

video.addEventListener('timeupdate', sync3DWithVideo);
video.addEventListener('seeked', sync3DWithVideo);

let globalTrajectoryData = null;

// Fetch the real data converted from the Parquet file
fetch('trajectory_data.json')
    .then(response => {
        if (!response.ok) throw new Error("trajectory_data.json not found. Did you run data_converter.py?");
        return response.json();
    })
    .then(data => {
        globalTrajectoryData = data;
        buildTrajectories(data);
    })
    .catch(err => console.error("Data Loading Error:", err));

function sync3DWithVideo() {
    if (!globalTrajectoryData) return; // Prevent crashing before data loads

    const currentFrame = Math.floor(video.currentTime * globalTrajectoryData.fps);
    frameDisplay.innerText = currentFrame;

    for (let i = 0; i < trajectoryLines.length; i++) {
        const lineObj = trajectoryLines[i];
        const dotObj = pointClouds[i];
        const path = lineObj.data;

        const targetIdx = Math.min(currentFrame, path.length - 1);

        if (targetIdx >= 0) {
            lineObj.line.geometry.setDrawRange(0, targetIdx);

            const currentPos = path[targetIdx];
            const positions = dotObj.dot.geometry.attributes.position.array;
            positions[0] = currentPos.x;
            positions[1] = currentPos.y;
            positions[2] = currentPos.z;
            dotObj.dot.geometry.attributes.position.needsUpdate = true;
        }
    }
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();