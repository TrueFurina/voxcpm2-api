// VoxCPM2 UI – Tauri backend
// The desktop app can auto-start a bundled API on macOS, while the frontend
// still talks HTTP/WS directly so it can also connect to an external API.

use std::fs;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager, RunEvent};

const LOCAL_API_HOST: &str = "127.0.0.1";
const LOCAL_API_PORT: u16 = 4000;
const LOCAL_API_BOOT_TIMEOUT: Duration = Duration::from_secs(12);

#[derive(Default)]
struct LocalApiState {
    child: Mutex<Option<Child>>,
}

/// Read any local file and return it as a base64 string.
/// Used to load reference / prompt audio files chosen via the file picker.
#[tauri::command]
fn read_file_as_base64(path: String) -> Result<String, String> {
    let bytes = fs::read(&path).map_err(|e| format!("Cannot read {path}: {e}"))?;
    Ok(base64_encode(&bytes))
}

/// Return just the filename part of a path (for display).
#[tauri::command]
fn file_name(path: String) -> String {
    Path::new(&path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Minimal base64 encoder – avoids pulling in the `base64` crate.
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 {
            chunk[1] as usize
        } else {
            0
        };
        let b2 = if chunk.len() > 2 {
            chunk[2] as usize
        } else {
            0
        };
        out.push(CHARS[b0 >> 2] as char);
        out.push(CHARS[((b0 & 3) << 4) | (b1 >> 4)] as char);
        if chunk.len() > 1 {
            out.push(CHARS[((b1 & 0xf) << 2) | (b2 >> 6)] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(CHARS[b2 & 0x3f] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn local_api_socket_addr() -> SocketAddr {
    format!("{LOCAL_API_HOST}:{LOCAL_API_PORT}")
        .parse()
        .expect("valid local API socket address")
}

fn local_api_log_path() -> PathBuf {
    std::env::temp_dir().join("voxcpm2-ui-local-api.log")
}

fn can_bind_local_api_port() -> bool {
    TcpListener::bind(local_api_socket_addr()).is_ok()
}

fn local_api_is_healthy() -> bool {
    let Ok(mut stream) =
        std::net::TcpStream::connect_timeout(&local_api_socket_addr(), Duration::from_millis(250))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }

    let mut response = [0_u8; 256];
    let Ok(bytes_read) = stream.read(&mut response) else {
        return false;
    };
    if bytes_read == 0 {
        return false;
    }

    let response = String::from_utf8_lossy(&response[..bytes_read]);
    response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200")
}

fn bundled_api_executable(app: &AppHandle) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir().ok()?;
    [
        resource_dir
            .join("api-macos")
            .join("voxcpm2-api")
            .join("voxcpm2-api"),
        resource_dir
            .join("resources")
            .join("api-macos")
            .join("voxcpm2-api")
            .join("voxcpm2-api"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

fn build_local_api_command(app: &AppHandle) -> Result<Command, String> {
    let executable = bundled_api_executable(app).unwrap_or_else(|| PathBuf::from("voxcpm2-api"));
    let log_path = local_api_log_path();
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|err| format!("cannot open API log at {}: {err}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .map_err(|err| format!("cannot clone API log handle: {err}"))?;

    let mut command = Command::new(&executable);
    command
        .env("VOXCPM2_API_HOST", LOCAL_API_HOST)
        .env("VOXCPM2_API_PORT", LOCAL_API_PORT.to_string())
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));

    if let Some(parent) = executable.parent().filter(|_| executable.is_absolute()) {
        command.current_dir(parent);
    }

    Ok(command)
}

fn wait_for_local_api(child: &mut Child) -> Result<(), String> {
    let deadline = Instant::now() + LOCAL_API_BOOT_TIMEOUT;
    while Instant::now() < deadline {
        if local_api_is_healthy() {
            return Ok(());
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("cannot poll local API process: {err}"))?
        {
            return Err(format!("local API exited early with status {status}"));
        }
        thread::sleep(Duration::from_millis(200));
    }
    Ok(())
}

fn start_local_api(app: &AppHandle) -> Result<(), String> {
    if !cfg!(target_os = "macos") {
        return Ok(());
    }

    if local_api_is_healthy() {
        return Ok(());
    }

    if !can_bind_local_api_port() {
        return Ok(());
    }

    let mut child = build_local_api_command(app)?
        .spawn()
        .map_err(|err| format!("failed to spawn local API: {err}"))?;
    wait_for_local_api(&mut child)?;

    let state = app.state::<LocalApiState>();
    *state
        .child
        .lock()
        .map_err(|_| "local API state lock poisoned".to_string())? = Some(child);
    Ok(())
}

fn stop_local_api(app: &AppHandle) {
    let state = app.state::<LocalApiState>();
    let Ok(mut guard) = state.child.lock() else {
        return;
    };
    let Some(mut child) = guard.take() else {
        return;
    };
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .manage(LocalApiState::default())
        .setup(|app| {
            if let Err(err) = start_local_api(&app.handle()) {
                eprintln!("local API startup failed: {err}");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![read_file_as_base64, file_name])
        .build(tauri::generate_context!())
        .expect("error while building VoxCPM2 UI");

    app.run(|app, event| {
        if matches!(event, RunEvent::Exit { .. }) {
            stop_local_api(app);
        }
    });
}
