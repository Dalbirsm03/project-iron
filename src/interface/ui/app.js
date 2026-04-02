// Global Scene Setup
const container = document.getElementById('canvas-3d');
const video = document.getElementById('video-player');
const frameDisplay = document.getElementById('frame-display');

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 2 / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth / 2, window.innerHeight);
container.appendChild(renderer.domElement);

camera.position.z = 500;

// Mock Data: Since the Python Parquet API isn't ready yet, use this mock data to prove the logic.
const mockData = {
    fps: 30,
    tracks: [
        { id: 1, path: Array.from({length: 300}, (_, i) => ({ f: i, x: i - 150, y: Math.sin(i/10)*50, z: Math.cos(i/10)*50 })) },
        { id: 2, path: Array.from({length: 300}, (_, i) => ({ f: i, x: (i - 150)*1.5, y: Math.cos(i/10)*50, z: Math.sin(i/10)*50 })) }
    ]
};

const trajectoryLines = [];
const pointClouds = [];

// ==========================================
// TASK 6.1: Implement Trajectory Trails
// ==========================================
function buildTrajectories(data) {
    data.tracks.forEach(track => {
        // Create LineSegments for the trail
        const points = [];
        track.path.forEach(pt => {
            points.push(new THREE.Vector3(pt.x, pt.y, pt.z));
        });

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5 });
        const line = new THREE.Line(geometry, material);
        
        // Hide line initially; will be revealed by time sync
        line.geometry.setDrawRange(0, 0); 
        scene.add(line);
        trajectoryLines.push({ line: line, data: track.path });

        // Create the moving Point Cloud marker
        const dotGeo = new THREE.BufferGeometry();
        dotGeo.setAttribute('position', new THREE.Float32BufferAttribute([0,0,0], 3));
        const dotMat = new THREE.PointsMaterial({ color: 0xff0000, size: 5 });
        const dot = new THREE.Points(dotGeo, dotMat);
        scene.add(dot);
        pointClouds.push({ dot: dot, data: track.path });
    });
}

// ==========================================
// TASK 6.2: Time Synchronization
// ==========================================
function sync3DWithVideo() {
    // Calculate exact frame based on video time and FPS
    const currentFrame = Math.floor(video.currentTime * mockData.fps);
    frameDisplay.innerText = currentFrame;

    for (let i = 0; i < trajectoryLines.length; i++) {
        const lineObj = trajectoryLines[i];
        const dotObj = pointClouds[i];
        const path = lineObj.data;

        // Ensure we don't exceed array bounds
        const targetIdx = Math.min(currentFrame, path.length - 1);

        if (targetIdx >= 0) {
            // Update Line Trail to only draw up to the current frame
            lineObj.line.geometry.setDrawRange(0, targetIdx);

            // Update Point Cloud position instantly to match the frame
            const currentPos = path[targetIdx];
            const positions = dotObj.dot.geometry.attributes.position.array;
            positions[0] = currentPos.x;
            positions[1] = currentPos.y;
            positions[2] = currentPos.z;
            dotObj.dot.geometry.attributes.position.needsUpdate = true;
        }
    }
    renderer.render(scene, camera);
}

// Attach the synchronization logic to the video player's time updates
video.addEventListener('timeupdate', sync3DWithVideo);
video.addEventListener('seeked', sync3DWithVideo); // Triggers when scrubbing the timeline

// Initialize
buildTrajectories(mockData);

// Animation Loop
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();