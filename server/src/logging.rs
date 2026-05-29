fn log_room_line(state: &AppState, room_id: &str, notation: &str) {
    let _ = fs::create_dir_all(state.log_root.as_ref());
    let path = state.log_root.join(format!("{room_id}.log"));
    println!("{room_id} {notation}");
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{notation}");
    }
}
