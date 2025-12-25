use bevy::prelude::*;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::math::primitives::RegularPolygon;
use bevy::utils::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::f32::consts::PI;
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

#[derive(Component)]
struct Predator;

#[derive(Component)]
struct Heading(f32);

/*
#[derive(Clone, Copy, Debug)]
struct BoidState {
    entity: Entity,
    pos: Vec2,
    vel: Vec2,
}
*/

/*
#[derive(Event, Clone, Copy)]
struct ApplyForceEvent {
    entity: Entity,
    force: Vec2,
}
*/

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

#[derive(Component)]
struct TrailDot {
    ttl: Timer,
}

#[derive(Resource)]
struct TrailSettings {
    spawn_every: Timer,
}

// Types and classes

impl Grid {
    fn new(cell_size: f32) -> Self {
        let mut cells: HashMap<IVec2, Vec<Entity>> = HashMap::default();
        cells.reserve(expected_entities / 2 + 1);
        Self { 
            cell_size, cells 
        }
    }

    fn clear(&mut self) {
        self.cells.clear();
    }

    fn cell_of(&self, pos: Vec2) -> IVec2 {
        (pos / self.cell_size).floor().as_ivec2()
    }

    fn insert(&mut self, pos: Vec2, e: Entity) {
        self.cells.entry(c).or_insert_with(|| {
            let mut v = Vec::new();
            v.reserve(8); // Reserve density
            v
        }).push(e);
    }

    fn candidates(&self, pos: Vec2, radius: f32, mut f: impl FnMut(Entity)) {
        let center = self.cell_of(pos);
        let r = (radius / self.cell_size).ceil() as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                let cell = center + IVec2::new(dx, dy);
                if let Some(list) = self.cells.get(&cell) {
                    for &e in list {
                        f(e);
                    }
                }
            }
        }
    }
}

// type Snapshot = Vec<BoidState>;

type BoidQueryItem<'a> = (
    Entity,
    &'a Transform,
    &'a Velocity,
    Option<&'a Predator>,
    &'a Heading
);

type BoidMutQueryItem<'a> = (
    &'a mut Transform,
    &'a mut Velocity,
    &'a mut Acceleration,
    &'a mut Heading,
);

type boidMutWriteItem<'a> = (
    Entity,
    &'a mut Acceleration,
);

// Constants

const EPS: f32 = 1e-5;

const NUM_BOIDS: usize = 150;
const NUM_PREDATORS: usize = 5;

const LEFT: f32 = -600.0;
const RIGHT: f32 = 600.0;
const TOP: f32 = 300.0;     
const BOTTOM: f32 = -300.0;

const TURN_FACTOR: f32 = 50.0;
const MAX_SPEED: f32 = 400.0;
const MIN_SPEED: f32 = 200.0;
const PREDATOR_SPEED: f32 = 520.0;

const PERCEPTION_RADIUS: f32 = 120.0;
const SEPARATION_RADIUS: f32 = 40.0;
const CHASE_RADIUS: f32 = 220.0;
const FLEE_RADIUS: f32 = 260.0;
const MAX_FORCE: f32 = 300.0;
const PREDATOR_FORCE: f32 = 420.0;

const TURN_SMOOTH: f32 = 12.0;
const BANK_FACTOR: f32 = 0.7;
const MAX_BANK: f32 = 0.6;

/*
const ALIGN_WEIGHT: f32 = 1.8;
const COHESION_WEIGHT: f32 = 1.2;
const SEPARATION_WEIGHT: f32 = 2.8;
*/

const CHASE_WEIGHT: f32 = 3.5;
const FLEE_WEIGHT: f32 = 5.0;

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
    } 
    else {
        v
    }
}

fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = b - a;
    while d > PI { d -= 2.0 * PI; }
    while d < -PI { d += 2.0 * PI; }
    d
}

fn build_grid_system(
    mut grid: ResMut<Grid>,
    boids: Query<BoidQueryItem<'_>, With<Boid>>,
) {
    grid.clear();
    for (et, tr, _, _, _) in boids.iter() {
        let pos = tr.translation.truncate();
        grid.insert(pos, et);
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
    let predator_mat = materials.add(Color::rgb(1.0, 0.2, 0.2));
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
            Heading(0.0),
        ));
    }
    for i in 0..NUM_PREDATORS {
        let x = (i as f32 * 200.0) - 200.0;
        let y = 0.0;
        commands.spawn((
            Boid,
            Predator,
            Velocity(Vec2::new(200.0, 0.0)),
            Acceleration(Vec2::ZERO),
            MaterialMesh2dBundle {
                mesh: msh.clone().into(),
                material: predator_mat.clone(),
                transform: Transform {
                    translation: Vec3::new(x, y, 0.0),
                    scale: Vec3::new(0.9, 2.2, 1.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            Heading(0.0)
        ));
    }

}

fn accumulate_neighbors<Acc>(
    entity_1: Entity,
    pos_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
    radius: f32,
    mut acc: Acc,
    mut accumulate: impl FnMut(&mut Acc, Vec2, Vec2, f32),
) -> (Acc, u32) {
    let r2 = radius * radius;
    let mut count: u32 = 0;

    grid.for_each_candidate(pos_1, radius, |e2| {
        if e2 == entity_1 {
            continue;
        }
        let Ok((_, tr_2, vel_2, _, _)) = boids.get(e2) 
        else {
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
    mut query: Query<BoidMutQueryItem<'_>, With<Boid>>
) {
    for (transform, mut velocity, _, _) in query.iter_mut() {
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
    mut query: Query<BoidMutQueryItem<'_>, With<Boid>>,
) {
    for (mut transform, mut velocity, mut acceleration, mut heading) in &mut query {
        let dt = time.delta_seconds();
        velocity.0 += acceleration.0 * dt;
        // NaN: calc velocity after clamping
        let speed = velocity.0.length();
        if speed > MAX_SPEED {
            velocity.0 = (velocity.0 / speed) * MAX_SPEED;
        }
        if speed < MIN_SPEED {
            velocity.0 = (velocity.0 / speed) * MIN_SPEED;
        }
        let v = velocity.0;
        if v.length_squared() > EPS {
            let target = v.y.atan2(v.x);
            let d = angle_diff(heading.0, target);
            heading.0 += d * (1.0 - (-TURN_SMOOTH * dt).exp());
            let bank = (d * BANK_FACTOR).clamp(-MAX_BANK, MAX_BANK);
            transform.rotation = Quat::from_rotation_z((heading.0 - FRAC_PI_2) + bank);
        }
        transform.translation += velocity.0.extend(0.0) * dt;
        acceleration.0 = Vec2::ZERO;
    }
}

fn separation_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
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
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
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
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
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

fn flee_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
) -> Vec2 {
    let mut away = Vec2::ZERO;
    let mut count = 0.0;
    let r2 = FLEE_RADIUS * FLEE_RADIUS;

    grid.for_each_candidate(pos_1, FLEE_RADIUS, |e2| {
        if e2 == entity_1 { 
            continue; 
        }
        let Ok((_, tr2, _, is_pred, _)) = boids.get(e2) 
        else { 
            continue; 
        };
        if is_pred.is_none() { 
            continue; 
        }
        let pos2 = tr2.translation.truncate();
        let d = pos_1 - pos2;
        let dist2 = d.length_squared();
        if dist2 < r2 && dist2 > EPS {
            away += d / dist2.sqrt();
            count += 1.0;
        }
    }
    if count == 0.0 { 
        return Vec2::ZERO; 
    }
    let desired = (away / count).normalize_or_zero() * MAX_SPEED;
    clamp_len(desired - vel_1, MAX_FORCE) * FLEE_WEIGHT
}

fn chase_force(
    entity_1: Entity,
    pos_1: Vec2,
    vel_1: Vec2,
    boids: &Query<BoidQueryItem<'_>, With<Boid>>,
    grid: &Grid,
) -> Vec2 {
    let mut best: Option<(f32, Vec2)> = None;
    let r2 = CHASE_RADIUS * CHASE_RADIUS;

    grid.for_each_candidate(pos_1, CHASE_RADIUS, |e2| {
        if e2 == entity_1 { 
            continue; 
        }
        let Ok((_, tr2, _, is_pred, _)) = boids.get(e2) 
        else { 
            continue; 
        };
        if is_pred.is_some() { 
            continue; 
        } 
        let pos2 = tr2.translation.truncate();
        let dist2 = (pos2 - pos_1).length_squared();
        if dist2 < r2 {
            match best {
                None => best = Some((dist2, pos2)),
                Some((bd2, _)) if dist2 < bd2 => best = Some((dist2, pos2)),
                _ => {}
            }
        }
    }
    let Some((_, target)) = best 
    else { 
        return Vec2::ZERO; 
    };
    let desired = (target - pos_1).normalize_or_zero() * PREDATOR_SPEED;
    clamp_len(desired - vel_1, PREDATOR_FORCE) * CHASE_WEIGHT
}

fn flocking_system(
    mut boids: ParamSet<(
        Query<BoidQueryItem<'_>, With<Boid>>,
        Query<boidMutWriteItem<'_>, With<Boid>>, 
    )>,
    grid: Res<Grid>,
    weights: Res<FlockingParams>,
) {
    let boids_p0 = boids.p0();
    for (et, tr, vel, pred, _) in boids_p0.iter() {
        let pos = tr.translation.truncate();
        let vel = vel.0;

        let force = 
        if pred.is_some() {
            chase_force(et, pos, vel, &boids, &grid)
        } 
        else {
            separation_force(et, pos, vel, &boids, &grid, &weights)
                + alignment_force(et, pos, vel, &boids, &grid, &weights)
                + cohesion_force(et, pos, vel, &boids, &grid, &weights)
                + flee_force(et, pos, vel, &boids, &grid)
        };
        if let Ok((_e, mut acc)) = boids.p1().get_mut(et) {
            acc.0 += force;
        }
    }
}

/*
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
*/

fn flocking_ui(mut contexts: EguiContexts, mut params: ResMut<FlockingParams>) {
    egui::Window::new("Flocking")
        .resizable(false)
        .show(contexts.ctx_mut(), |ui| {
            ui.add(egui::Slider::new(&mut params.alignment, 0.0..=5.0).text("Alignment"));
            ui.add(egui::Slider::new(&mut params.cohesion, 0.0..=5.0).text("Cohesion"));
            ui.add(egui::Slider::new(&mut params.separation, 0.0..=5.0).text("Separation"));
        });
}

fn trail_spawn_system(
    time: Res<Time>,
    mut settings: ResMut<TrailSettings>,
    boids: Query<&Transform, With<Boid>>,
    mut commands: Commands,
) {
    settings.spawn_every.tick(time.delta());
    if !settings.spawn_every.finished() { return; }

    for tr in boids.iter() {
        commands.spawn((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::rgba(1.0, 1.0, 1.0, 0.25),
                    custom_size: Some(Vec2::splat(2.0)),
                    ..Default::default()
                },
                transform: Transform::from_translation(tr.translation),
                ..Default::default()
            },
            TrailDot {
                ttl: Timer::from_seconds(0.6, TimerMode::Once),
            },
        ));
    }
}

fn trail_cleanup_system(
    time: Res<Time>,
    mut commands: Commands,
    mut dots: Query<(Entity, &mut TrailDot)>,
) {
    for (e, mut dot) in dots.iter_mut() {
        dot.ttl.tick(time.delta());
        if dot.ttl.finished() {
            commands.entity(e).despawn();
        }
    }
}

fn main() {
    App::new()
        //.add_event::<ApplyForceEvent>()
        .insert_resource(Grid::new(PERCEPTION_RADIUS, NUM_BOIDS + NUM_PREDATORS))
        .insert_resource(FlockingParams {
            alignment: 1.0,
            cohesion: 0.8,
            separation: 1.4,
        })
        .insert_resource(TrailSettings {
            spawn_every: Timer::from_seconds(0.05, TimerMode::Repeating),
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (
            flocking_ui,
            build_grid_system,
            boundary_system,
            flocking_system,
            integrate_system,
            trail_spawn_system,
            trail_cleanup_system,
        ))
        .run();
}