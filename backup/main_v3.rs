use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::math::primitives::RegularPolygon;
use bevy::utils::HashMap;
use std::f32::consts::FRAC_PI_2;
use bevy_egui::{egui, EguiContexts, EguiPlugin};

// Components 

#[derive(Component)]
struct Boid;

#[derive(Component)]
struct BoidCamera;

#[derive(Component)]
struct Velocity(Vec2);

#[derive(Component)]
struct Acceleration(Vec2);

/*
#[derive(Clone, Copy, Debug)]
struct BoidState {
    entity: Entity,
    pos: Vec2,
    vel: Vec2,
}
*/

#[derive(Event, Clone, Copy)]
struct ApplyForceEvent {
    entity: Entity,
    force: Vec2,
}

#[derive(Resource)]
struct Grid {
    cell_size: f32,
    cells: HashMap<IVec2, Vec<Entity>>,
}

#[derive(Resource)]
struct FlockingParams {
    alignment: f32,
    cohesion: f32,
    separation: f32,
}

// Grid class

impl Grid {
    fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn clear(&mut self) {
        self.cells.clear();
    }

    fn cell_of(&self, pos: Vec2) -> IVec2 {
        (pos / self.cell_size).floor().as_ivec2()
    }

    fn insert(&mut self, pos: Vec2, e: Entity) {
        let c = self.cell_of(pos);
        self.cells.entry(c).or_default().push(e);
    }

    fn candidates<'a>(&'a self, pos: Vec2, radius: f32) -> impl Iterator<Item = Entity> + 'a {
        let center = self.cell_of(pos);

        let r = (radius / self.cell_size).ceil() as i32;

        (-r..=r).flat_map(move |dy| {
            (-r..=r).flat_map(move |dx| {
                let cell = center + IVec2::new(dx, dy);
                self.cells
                    .get(&cell)
                    .into_iter()
                    .flat_map(|v| v.iter().copied())
            })
        })
    }
}

// type Snapshot = Vec<BoidState>;

// Constants

const LEFT: f32 = -600.0;
const RIGHT: f32 = 600.0;
const TOP: f32 = 300.0;
const BOTTOM: f32 = -300.0;

const TURN_FACTOR: f32 = 50.0;
const MAX_SPEED: f32 = 400.0;
const MIN_SPEED: f32 = 200.0;

const NUM_BOIDS: usize = 120;

const PERCEPTION_RADIUS: f32 = 120.0;
const SEPARATION_RADIUS: f32 = 40.0;
const MAX_FORCE: f32 = 300.0;

/*
const ALIGN_WEIGHT: f32 = 1.8;
const COHESION_WEIGHT: f32 = 1.2;
const SEPARATION_WEIGHT: f32 = 2.8;
*/

const EPS: f32 = 1e-5;

/*
fn snapshot_boids(query: &Query<(Entity, &Transform, &Velocity), With<Boid>>) -> Snapshot {
    query
        .iter()
        .map(|(entity, t, v)| BoidState {
            entity,
            pos: t.translation.truncate(),
            vel: v.0,
        })
        .collect()
}
*/

// Helper functions 

fn clamp_len(v: Vec2, max: f32) -> Vec2 {
    let len = v.length();
    if len > max && len > EPS {
        v / len * max
    } else {
        v
    }
}

fn build_grid_system(
    mut grid: ResMut<Grid>,
    boids: Query<(Entity, &Transform), With<Boid>>,
) {
    grid.clear();
    for (e, tr) in boids.iter() {
        let pos = tr.translation.truncate();
        grid.insert(pos, e);
    }
}

// Main functions

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        Camera2dBundle::default(),
        BoidCamera,
    ));
    let triangle = Mesh::from(RegularPolygon::new(10.0, 3));
    let msh = meshes.add(triangle);
    let mat = materials.add(Color::WHITE);
    for i in 0..NUM_BOIDS {
        let x = ((i as f32) * 37.0).sin() * 300.0;
        let y = ((i as f32) * 53.0).cos() * 200.0;
        let vx = ((i as f32) * 11.0).cos() * 120.0;
        let vy = ((i as f32) * 17.0).sin() * 120.0;

        commands.spawn((
            Boid,
            Velocity(Vec2::new(vx, vy)),
            Acceleration(Vec2::ZERO),
            MaterialMesh2dBundle {
                mesh: msh.clone().into(),
                material: mat.clone(),
                transform: Transform {
                    translation: Vec3::new(x, y, 0.0),
                    scale: Vec3::new(0.6, 1.6, 1.0), 
                    ..Default::default()
                },
                ..Default::default()
            },
        ));
    }
}

fn accumulate_neighbors<Acc>(
    entity_1: Entity,
    pos_1: Vec2,
    boids: &Query<(Entity, &Transform, &Velocity), With<Boid>>,
    grid: &Grid,
    radius: f32,
    mut acc: Acc,
    mut accumulate: impl FnMut(&mut Acc, Vec2, Vec2, f32),
) -> (Acc, u32) {
    let r2 = radius * radius;
    let mut count: u32 = 0;

    for entity_2 in grid.candidates(pos_1, radius) {
        if entity_2 == entity_1 {
            continue;
        }
        let Ok((_, tr_2, vel_2)) = boids.get(entity_2) else {
            continue;
        };
        let pos_2 = tr_2.translation.truncate();
        let d = pos_2 - pos_1;
        let dist2 = d.length_squared();

        if dist2 < r2 {
            accumulate(&mut acc, pos_2, vel_2.0, dist2);
            count += 1;
        }
    }

    (acc, count)
}

fn boundary_system(
    mut query: Query<(&mut Transform, &mut Velocity), 
    With<Boid>>
) {
    for (transform, mut velocity) in query.iter_mut() {
        let pos = transform.translation;

        if pos.x < LEFT {
            velocity.0.x += TURN_FACTOR;
        } 
        else if pos.x > RIGHT {
            velocity.0.x -= TURN_FACTOR;
        }
        if pos.y < BOTTOM {
            velocity.0.y += TURN_FACTOR;
        } 
        else if pos.y > TOP {
            velocity.0.y -= TURN_FACTOR;
        }
    }
}

fn integrate_system(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut Velocity, &mut Acceleration), 
    With<Boid>>,
) {
    for (mut transform, mut velocity, mut acceleration) in &mut query {
        velocity.0 += acceleration.0 * time.delta_seconds();
        let speed = velocity.0.length();
        if speed > MAX_SPEED {
            velocity.0 = (velocity.0 / speed) * MAX_SPEED;
        }
        if speed < MIN_SPEED {
            velocity.0 = (velocity.0 / speed) * MIN_SPEED;
        }
        if velocity.0.length_squared() > EPS {
            let angle = velocity.0.y.atan2(velocity.0.x);
            transform.rotation = Quat::from_rotation_z(angle - FRAC_PI_2);
        }
        transform.translation += velocity.0.extend(0.0) * time.delta_seconds();
        acceleration.0 = Vec2::ZERO;
    }
}

fn separation_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<(Entity, &Transform, &Velocity), With<Boid>>,
    grid: &Grid,
    weights: &FlockingParams,
) -> Vec2 {
    let (push_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
        grid,
        SEPARATION_RADIUS,
        Vec2::ZERO,
        |acc, pos_2, _, dist2| {
            let offset = pos_1 - pos_2;
            if dist2 > EPS {
                *acc += offset / dist2.sqrt();
            }
        },
    );

    if count == 0 {
        return Vec2::ZERO;
    }

    let push = (push_sum / count as f32).normalize_or_zero();
    let force = push * MAX_SPEED;
    clamp_len(force - vel_1, MAX_FORCE) * weights.separation
}

fn alignment_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<(Entity, &Transform, &Velocity), With<Boid>>,
    grid: &Grid,
    weights: &FlockingParams,
) -> Vec2 {
    let (vel_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
        grid,
        PERCEPTION_RADIUS,
        Vec2::ZERO,
        |acc, _, vel_2, _| {
            *acc += vel_2;
        },
    );

    if count == 0 {
        return Vec2::ZERO;
    }

    let avg_vel = (vel_sum / count as f32).normalize_or_zero();
    let force = avg_vel * MAX_SPEED;
    clamp_len(force - vel_1, MAX_FORCE) * weights.alignment
}

fn cohesion_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<(Entity, &Transform, &Velocity), With<Boid>>,
    grid: &Grid,
    weights: &FlockingParams,
) -> Vec2 {
    let (pos_sum, count) = accumulate_neighbors(
        entity_1,
        pos_1,
        boids,
        grid,
        PERCEPTION_RADIUS,
        Vec2::ZERO,
        |acc, pos_2, _, _| {
            *acc += pos_2;
        },
    );

    if count == 0 {
        return Vec2::ZERO;
    }

    let center = pos_sum / count as f32;
    let delta = (center - pos_1).normalize_or_zero();
    let force = delta * MAX_SPEED;
    clamp_len(force - vel_1, MAX_FORCE) * weights.cohesion
}

fn flocking_system(
    boids: Query<(Entity, &Transform, &Velocity), With<Boid>>,
    mut writer: EventWriter<ApplyForceEvent>,
    grid: Res<Grid>,
    weights: Res<FlockingParams>,
) {
    for (et, tr, vel) in boids.iter() {
        let pos = tr.translation.truncate();
        let vel = vel.0;

        let force =
            separation_force(et, pos, vel, &boids, &grid, &weights)
            + alignment_force(et, pos, vel, &boids, &grid, &weights)
            + cohesion_force(et, pos, vel, &boids, &grid, &weights);

        writer.send(ApplyForceEvent { entity: et, force });
    }
}

fn apply_forces_system(
    mut reader: EventReader<ApplyForceEvent>,
    mut query: Query<&mut Acceleration, With<Boid>>,
) {
    for ev in reader.read() {
        if let Ok(mut acc) = query.get_mut(ev.entity) {
            acc.0 += ev.force;
        }
    }
}

fn flocking_ui(mut contexts: EguiContexts, mut params: ResMut<FlockingParams>) {
    egui::Window::new("Flocking")
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| {
            ui.add(egui::Slider::new(&mut params.alignment, 0.0..=5.0).text("Alignment"));
            ui.add(egui::Slider::new(&mut params.cohesion, 0.0..=5.0).text("Cohesion"));
            ui.add(egui::Slider::new(&mut params.separation, 0.0..=5.0).text("Separation"));
        });
}

fn main() {
    App::new()
        .add_event::<ApplyForceEvent>()
        .insert_resource(Grid::new(PERCEPTION_RADIUS))
        .insert_resource(FlockingParams {
            alignment: 1.0,
            cohesion: 0.8,
            separation: 1.4,
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (
            flocking_ui,
            build_grid_system,
            boundary_system,
            integrate_system,
            flocking_system,
            apply_forces_system,
        ))
        .run();
}