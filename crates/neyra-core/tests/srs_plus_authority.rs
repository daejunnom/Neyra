//! Compare actual rotations against independently extracted source geometry
//! and ordered attempts. These cases stay below the ceiling so this audit is
//! independent of Neyra's separate board-height search policy.

use neyra_core::physics::{Board, Input, Piece, Pose, HEIGHT, WIDTH};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct Fixture {
    transitions_y_up: BTreeMap<String, Vec<[i8; 2]>>,
    standard_geometry_y_down: BTreeMap<String, Geometry>,
}

#[derive(Deserialize)]
struct Geometry {
    cells: [[[i8; 2]; 4]; 4],
}

fn piece_name(piece: Piece) -> &'static str {
    match piece {
        Piece::I => "I",
        Piece::J => "J",
        Piece::L => "L",
        Piece::O => "O",
        Piece::S => "S",
        Piece::T => "T",
        Piece::Z => "Z",
    }
}

// Deliberately use fixture geometry rather than Pose::cells or Board::legal:
// an axis/origin/shape error in the engine must not be shared by the oracle.
fn fixture_pose_is_free(board: &Board, geometry: &Geometry, pose: Pose) -> bool {
    geometry.cells[pose.rotation as usize]
        .iter()
        .all(|&[dx, dy]| {
            let (x, y) = (pose.x + dx, pose.y - dy);
            x >= 0
                && x < WIDTH as i8
                && y >= 0
                && y < HEIGHT as i8
                && board.rows[y as usize] & (1 << x) == 0
        })
}

#[test]
fn rotations_match_the_official_fixture_on_walls_floors_and_overhangs() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "fixtures/tetrio_srs_plus_standard_tetromino_kicks.json"
    ))
    .unwrap();
    assert_eq!(fixture.transitions_y_up.len(), 84);
    let terrain: &[&[u16]] = &[
        &[],
        &[23, 600, 993, 917, 147, 849, 763],
        &[0b1111001111, 0b0011001100, 0b0000001000],
        &[0b1110011111, 0b1100001111, 0b1000011111, 0b1000000011],
        &[0b1010101010, 0b0100010001, 0b0001000100, 0b0010000000],
    ];
    let mut covered = BTreeSet::new();
    let mut non_origin_kicks = 0;
    let mut blocked_rotations = 0;

    for piece in Piece::ALL {
        let name = piece_name(piece);
        let geometry = &fixture.standard_geometry_y_down[name];
        for from in 0..4 {
            let mut actual_cells = Pose {
                piece,
                x: 0,
                y: 0,
                rotation: from,
            }
            .cells();
            let mut expected_cells = geometry.cells[from as usize].map(|[x, y]| (x, -y));
            actual_cells.sort_unstable();
            expected_cells.sort_unstable();
            assert_eq!(actual_cells, expected_cells, "{name} rotation {from}");

            for rows in terrain {
                let mut board = Board::default();
                board.rows[..rows.len()].copy_from_slice(rows);
                for x in -3..WIDTH as i8 {
                    for y in 0..=12 {
                        let pose = Pose {
                            piece,
                            x,
                            y,
                            rotation: from,
                        };
                        if !fixture_pose_is_free(&board, geometry, pose) {
                            continue;
                        }
                        assert!(board.legal(pose));
                        for (input, delta) in [(Input::Cw, 1), (Input::Half, 2), (Input::Ccw, 3)] {
                            let to = (from + delta) & 3;
                            let key = format!("{name}:{from}{to}");
                            covered.insert(key.clone());
                            let expected = fixture.transitions_y_up[&key]
                                .iter()
                                .enumerate()
                                .find_map(|(index, &[dx, dy])| {
                                    let target = Pose {
                                        x: x + dx,
                                        y: y + dy,
                                        rotation: to,
                                        ..pose
                                    };
                                    fixture_pose_is_free(&board, geometry, target)
                                        .then_some((index, target))
                                });
                            match expected {
                                Some((index, _)) if index != 0 => non_origin_kicks += 1,
                                None => blocked_rotations += 1,
                                _ => {}
                            }
                            assert_eq!(
                                board.rotate(pose, input),
                                expected.map(|(_, target)| target),
                                "{key} from ({x}, {y}) on {rows:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        covered.len(),
        84,
        "every source transition must be exercised"
    );
    assert!(non_origin_kicks > 0, "exercise blocked origin attempts");
    assert!(blocked_rotations > 0, "exercise fully blocked rotations");
}
